"""Reset-safe gait evidence with force hysteresis and qualified landings."""
import torch
from ry_unitree_rl_lab.utils.balance_acceptance import assess_balance

class GaitMetrics:
    def __init__(self, raw, robot, foot_ids, sensor, sensor_ids, wm, scanner):
        self.raw,self.robot,self.feet,self.sensor,self.ids,self.wm,self.scanner=raw,robot,foot_ids,sensor,sensor_ids,wm,scanner
        n,d=raw.num_envs,raw.device
        self.front_ids, _ = sensor.find_bodies(['FL_foot','FR_foot'], preserve_order=True)
        self.stage2 = getattr(raw.cfg, "stage2_support", False)
        if self.stage2:
            self.nonfoot_ids = [i for i, name in enumerate(sensor.body_names)
                                if name not in ("RL_foot", "RR_foot")]
        self.nonfoot_failure_counts = {}
        self.active=torch.ones(n,dtype=torch.bool,device=d)
        self.count=torch.zeros(n,device=d)
        self.sums={k:torch.zeros(n,device=d) for k in ['speed','upright','single_support','flight','height','stance_slip_speed','lateral_speed_abs','vertical_speed_abs','front_contact','speed_error','horizontal_speed','yaw_rate_abs']}
        if self.stage2:
            self.sums.update({k:torch.zeros(n,device=d) for k in
                ["target_height_fraction","rear_support_fraction","nonfoot_contact_fraction"]})
        self.start=robot.data.root_pos_w[:,:2].clone(); self.end=self.start.clone()
        self.max_horizontal_displacement=torch.zeros(n,device=d)
        self.cycle_start=self.start.clone()
        self.cycle_displacements=[]
        self.cycle_steps=round(1.6/raw.step_dt)
        self.min_landing_height=.008 if raw.cfg.stage=='balance' else .02
        yaw=wm.upright_heading(raw); self.forward=torch.stack((torch.cos(yaw),torch.sin(yaw)),-1)
        self.previous_yaw=yaw.clone()
        self.heading_change=torch.zeros(n,device=d)
        self.max_heading_excursion=self.heading_change.clone()
        self.absolute_heading_travel=self.heading_change.clone()
        self.horizontal_path_length=torch.zeros(n,device=d)
        self.contact=torch.ones(n,2,dtype=torch.bool,device=d)
        self.air=torch.zeros(n,2,device=d); self.peak=self.air.clone(); self.duration=self.air.clone()
        self.landing_air=self.air.clone(); self.landing_peak=self.air.clone()
        self.pending=torch.zeros_like(self.contact)
        self.last_time=torch.full_like(self.air,-1)
        self.touch=torch.zeros(n,2,dtype=torch.long,device=d); self.raw_touch=self.touch.clone()
        self.last_foot=torch.full((n,),-1,dtype=torch.long,device=d)
        self.events=torch.zeros(n,dtype=torch.long,device=d); self.alternations=self.events.clone(); self.resets=self.events.clone()
        self.max_height=self.air.clone(); self.falls=0; self.terminations={}; self.trace=[]; self.landings=[]

    def update(self, step, done):
        r,robot=self.raw,self.robot; dt=r.step_dt; now=(step+1)*dt
        done=done.bool(); valid=self.active & ~done
        self.resets+=(r.termination_manager.terminated & self.active).long()
        self.falls+=int((r.termination_manager.terminated & self.active).sum())
        for name in r.termination_manager.active_terms:
            self.terminations[name]=self.terminations.get(name,0)+int((r.termination_manager.get_term(name)&self.active).sum())
        if self.stage2 and hasattr(r, "_stage2_nonfoot_peak_forces"):
            exceeded = r._stage2_nonfoot_peak_forces > r.cfg.nonfoot_contact_threshold
            failed = exceeded & (r.termination_manager.terminated & self.active)[:, None]
            for index, body_id in enumerate(self.nonfoot_ids):
                name = self.sensor.body_names[body_id]
                self.nonfoot_failure_counts[name] = self.nonfoot_failure_counts.get(name, 0) + int(failed[:, index].sum())
        force=self.sensor.data.net_forces_w[:,self.ids,2].clamp(min=0)
        old=self.contact.clone(); self.contact=torch.where(old,force>2.,force>5.)
        z=robot.data.body_pos_w[:,self.feet,2]; h=z-z.min(-1,keepdim=True).values
        self.max_height=torch.maximum(self.max_height,h*valid[:,None])
        landing=self.contact & ~old & valid[:,None]; self.raw_touch+=landing.long()
        self.landing_air=torch.where(landing,self.air,self.landing_air)
        self.landing_peak=torch.where(landing,self.peak,self.landing_peak)
        self.pending=(self.pending|landing)&self.contact&valid[:,None]
        self.duration=torch.where(self.contact,self.duration+dt,0.)
        self.air=torch.where(~self.contact,self.air+dt,0.)
        self.peak=torch.where(~self.contact,torch.maximum(self.peak,h),0.)
        event=self.pending&(self.duration>=0.04)&(self.landing_air>=0.06)&(self.landing_peak>=self.min_landing_height)&((now-self.last_time)>=0.18)
        simultaneous=event.all(-1); event &= ~simultaneous[:,None]
        for foot in range(2):
            e=event[:,foot]; self.alternations+=(e&(self.last_foot>=0)&(self.last_foot!=foot)).long()
            self.events+=e.long(); self.last_foot=torch.where(e,foot,self.last_foot)
            if bool(e[0]): self.landings.append(dict(time_s=now,foot=foot,air_s=float(self.landing_air[0,foot]),peak_m=float(self.landing_peak[0,foot])))
        self.touch+=event.long(); self.last_time=torch.where(event,now,self.last_time); self.pending &= ~event
        yaw=self.wm.upright_heading(r); vel=self.wm.heading_lin_vel(r); height=self.wm.root_height(r,self.scanner)
        upright=-robot.data.projected_gravity_b[:,0]
        command=r.command_manager.get_command('base_velocity')[:,0]
        front_contact=(self.sensor.data.net_forces_w[:,self.front_ids].norm(dim=-1)>1.0).any(dim=-1)

        slip=robot.data.body_lin_vel_w[:,self.feet,:2].norm(dim=-1)
        values=dict(speed=vel[:,0],front_contact=front_contact.float(),speed_error=(vel[:,0]-command).abs(),upright=((upright>0.94)&(height>0.43)).float(),
                    single_support=(self.contact.sum(-1)==1).float(),flight=(self.contact.sum(-1)==0).float(),height=height,
                    stance_slip_speed=(slip*self.contact).sum(-1)/self.contact.sum(-1).clamp(min=1),
                    lateral_speed_abs=vel[:,1].abs(),vertical_speed_abs=vel[:,2].abs(),
                    horizontal_speed=robot.data.root_lin_vel_w[:,:2].norm(dim=-1),
                    yaw_rate_abs=robot.data.root_ang_vel_w[:,2].abs())
        if self.stage2:
            all_force = self.sensor.data.net_forces_w
            other_load = all_force[:,self.nonfoot_ids,2].clamp(min=0.).sum(-1)
            rear_load = force.sum(-1)
            history = self.sensor.data.net_forces_w_history[:,:,self.nonfoot_ids]
            values.update(target_height_fraction=((height>=.49)&(height<=.55)).float(),
                          rear_support_fraction=rear_load/(rear_load+other_load).clamp(min=1e-6),
                          nonfoot_contact_fraction=(history.norm(dim=-1)>self.raw.cfg.nonfoot_contact_threshold).any(-1).any(-1).float())
        for key,value in values.items(): self.sums[key]+=value*valid
        self.count+=valid.float()
        next_position=torch.where(valid[:,None],robot.data.root_pos_w[:,:2],self.end)
        self.horizontal_path_length+=(next_position-self.end).norm(dim=-1)
        self.end=next_position
        delta=yaw-self.previous_yaw
        delta=torch.where(valid,torch.atan2(torch.sin(delta),torch.cos(delta)),0.)
        self.heading_change+=delta
        self.absolute_heading_travel+=delta.abs()
        self.max_heading_excursion=torch.maximum(self.max_heading_excursion,self.heading_change.abs())
        self.previous_yaw=torch.where(valid,yaw,self.previous_yaw)
        distance=(self.end-self.start).norm(dim=-1)
        self.max_horizontal_displacement=torch.maximum(self.max_horizontal_displacement,distance)
        if (step+1)%self.cycle_steps==0:
            displacement=(self.end-self.cycle_start).norm(dim=-1)
            self.cycle_displacements.append(dict(time_s=now,displacement_m=displacement.tolist(),valid=valid.tolist()))
            self.cycle_start=self.end.clone()
        self.active &= ~done
        if step%5==0:
            self.trace.append(dict(time_s=now,active=bool(valid[0]),position=robot.data.root_pos_w[0].tolist(),
                                   speed_mps=float(vel[0,0]),command_mps=float(r.command_manager.get_command('base_velocity')[0,0]),
                                   heading_yaw_rad=float(yaw[0]),world_lin_vel_mps=robot.data.root_lin_vel_w[0].tolist(),
                                   yaw_rate_world_radps=float(robot.data.root_ang_vel_w[0,2]),
                                   upright_cos=float(upright[0]),contact=self.contact[0].tolist(),force_n=force[0].tolist(),height_m=h[0].tolist()))

    def result(self, has_policy):
        per={key:value/self.count.clamp(min=1) for key,value in self.sums.items()}
        seconds=self.count*self.raw.step_dt
        displacement=((self.end-self.start)*self.forward).sum(-1)
        alternate=self.alternations.float()/(self.events-1).clamp(min=1)
        passed=((seconds>=15)&(self.resets==0)&(per['upright']>0.95)&(per['speed_error']<=0.04)&(per['front_contact']<0.01)&(per['flight']<0.1)
                &(per['single_support']>0.15)&(per['single_support']<0.9)&(self.touch.min(-1).values>=4)
                &(displacement>(self.raw.command_manager.get_command('base_velocity')[:,0]-0.04).clamp(min=0)*seconds)&(alternate>0.8)&(per['stance_slip_speed']<0.12))
        if self.stage2:
            passed &= ((per["target_height_fraction"]>.95)&(per["rear_support_fraction"]>.95)
                       &(per["nonfoot_contact_fraction"]==0.))
        path_speed=self.horizontal_path_length/seconds.clamp(min=self.raw.step_dt)
        balance_reports=[assess_balance(dict(
            valid_seconds=float(seconds[i]),failures=int(self.resets[i]),
            upright_fraction=float(per['upright'][i]),front_contact_fraction=float(per['front_contact'][i]),
            forward_error_mps=float(per['speed_error'][i]),mean_abs_yaw_rate_radps=float(per['yaw_rate_abs'][i]),
            max_heading_excursion_rad=float(self.max_heading_excursion[i]),
            max_horizontal_displacement_m=float(self.max_horizontal_displacement[i]),
            path_speed_mps=float(path_speed[i])),has_policy and self.raw.cfg.stage=='balance')
            for i in range(self.raw.num_envs)]
        if self.stage2:
            for i, report in enumerate(balance_reports):
                report["checks"].update(
                    target_height=bool(per["target_height_fraction"][i]>.95),
                    foot_support=bool(per["rear_support_fraction"][i]>.95),
                    no_nonfoot_contact=bool(per["nonfoot_contact_fraction"][i]==0.))
                report["failed_checks"]=[k for k,v in report["checks"].items() if not v]
                report["passed"]=all(report["checks"].values())
        result={key:float(value.mean()) for key,value in per.items()}
        result.update(nonfoot_failure_counts=self.nonfoot_failure_counts,falls=self.falls,resets_total=int(self.resets.sum()),termination_counts=self.terminations,
                      touchdowns_mean=self.touch.float().mean(0).tolist(),raw_touchdowns_mean=self.raw_touch.float().mean(0).tolist(),
                      qualified_touchdowns_per_env=self.touch.tolist(),valid_seconds_per_env=seconds.tolist(),
                      net_forward_displacement_m=displacement.tolist(),alternation_fraction_per_env=alternate.tolist(),
                      net_horizontal_displacement_m=(self.end-self.start).norm(dim=-1).tolist(),
                      max_horizontal_displacement_m=self.max_horizontal_displacement.tolist(),
                      horizontal_path_length_m=self.horizontal_path_length.tolist(),
                      path_speed_mps=path_speed.tolist(),
                      net_heading_change_rad=self.heading_change.tolist(),
                      max_heading_excursion_rad=self.max_heading_excursion.tolist(),
                      absolute_heading_travel_rad=self.absolute_heading_travel.tolist(),
                      cycle_duration_s=self.cycle_steps*self.raw.step_dt,
                      cycle_displacements=self.cycle_displacements,
                      landing_thresholds=dict(min_air_s=.06,min_lift_m=self.min_landing_height,min_support_s=.04),
                      alternating_steps_per_env=((self.touch.min(-1).values>=4)&(alternate>.8)).tolist(),
                      max_relative_foot_height=self.max_height.amax(0).tolist(),walking_acceptance_per_env=passed.tolist(),
                      walking_acceptance_passed=bool(has_policy and self.raw.cfg.stage=='walk' and passed.all()),
                      balance_acceptance_per_env=[v['passed'] for v in balance_reports],
                      balance_acceptance_reports=balance_reports,
                      balance_acceptance_passed=all(v['passed'] for v in balance_reports))
        return result
