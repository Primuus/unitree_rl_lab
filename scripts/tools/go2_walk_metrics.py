"""Reset-safe gait evidence with force hysteresis and qualified landings."""
import torch

class GaitMetrics:
    def __init__(self, raw, robot, foot_ids, sensor, sensor_ids, wm, scanner):
        self.raw,self.robot,self.feet,self.sensor,self.ids,self.wm,self.scanner=raw,robot,foot_ids,sensor,sensor_ids,wm,scanner
        n,d=raw.num_envs,raw.device
        self.front_ids, _ = sensor.find_bodies(['FL_foot','FR_foot'], preserve_order=True)
        self.active=torch.ones(n,dtype=torch.bool,device=d)
        self.count=torch.zeros(n,device=d)
        self.sums={k:torch.zeros(n,device=d) for k in ['speed','upright','single_support','flight','height','stance_slip_speed','lateral_speed_abs','vertical_speed_abs','front_contact','speed_error']}
        self.start=robot.data.root_pos_w[:,:2].clone(); self.end=self.start.clone()
        yaw=wm.upright_heading(raw); self.forward=torch.stack((torch.cos(yaw),torch.sin(yaw)),-1)
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
        event=self.pending&(self.duration>=0.04)&(self.landing_air>=0.06)&(self.landing_peak>=0.02)&((now-self.last_time)>=0.18)
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
                    lateral_speed_abs=vel[:,1].abs(),vertical_speed_abs=vel[:,2].abs())
        for key,value in values.items(): self.sums[key]+=value*valid
        self.count+=valid.float(); self.end=torch.where(valid[:,None],robot.data.root_pos_w[:,:2],self.end)
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
        balance_passed=(seconds>=15)&(self.resets==0)&(per['upright']>0.95)&(per['front_contact']<0.01)&(per['speed_error']<=0.04)
        result={key:float(value.mean()) for key,value in per.items()}
        result.update(falls=self.falls,resets_total=int(self.resets.sum()),termination_counts=self.terminations,
                      touchdowns_mean=self.touch.float().mean(0).tolist(),raw_touchdowns_mean=self.raw_touch.float().mean(0).tolist(),
                      qualified_touchdowns_per_env=self.touch.tolist(),valid_seconds_per_env=seconds.tolist(),
                      net_forward_displacement_m=displacement.tolist(),alternation_fraction_per_env=alternate.tolist(),
                      max_relative_foot_height=self.max_height.amax(0).tolist(),walking_acceptance_per_env=passed.tolist(),
                      walking_acceptance_passed=bool(has_policy and self.raw.cfg.stage=='walk' and passed.all()),
                      balance_acceptance_passed=bool(has_policy and self.raw.cfg.stage=='balance' and balance_passed.all()))
        return result
