package GSQuad
  package Components
    model Quadrotor
      // multi-fidelity quadrotor model
      // load packages
      import GSQuad.Constants;
      import GSQuad.Utils.quat2eul;
      import GSQuad.Utils.clip;
      // setup fidelity level and load different fidelity quadrotor model
      parameter Integer fidelity = 1;
      QuadLowFidelity quad_low;
      // parameters
      // [sec] PWM signal frequency
      parameter Real actuator_sample_period = 0.005 if fidelity == 1;
      // [sec] sensing frequency
      parameter Real sensor_sample_period = 0.005;
        // targeted acoustic attack model and parameters
      parameter Real W = 0;                                     // [W] power of speaker
      parameter Real dist = 0.01;                               // [m] distance to speaker
      parameter Real psi_ac = 80.0*Constants.d2r;               // [rad] speaker direction
      parameter Real w_ac = 15.0002e+3*2*Constants.pi;          // [rad/s] acoustic attack frequency
      parameter Real epsilon = 0.0*Constants.d2r;               // [rad] misalignment of gyroscope, reference - 1deg
      parameter Real phi_0 = 30*Constants.d2r;                      // [rad] phase shift for acoustic noise compared to driving signal
      GyroAcousticAtk gyroatk(W=W, dist=dist, psi_ac=psi_ac, w_ac=w_ac, epsilon=epsilon, phi_0=phi_0);
      // [-] minimum/maximum PWM
      parameter Real pwm_min = 1000;
      parameter Real pwm_max = 2000;
      // input: four PWM inputs for rotors
      // channel 1: right-up (ccw, pwm: [1000 2000])
      // channel 2: right-down (cw, pwm: [1000 2000])
      // channel 3: left-down (ccw, pwm: [1000 2000])
      // channel 4: left-up (cw, pwm: [1000 2000])
      Connectors.ControlBus control(nu=4) annotation(
        Placement(transformation(origin = {-120, 5}, extent = {{-19.8, -12.375}, {19.8, 12.375}}), iconTransformation(origin = {-129.4, 1.125}, extent = {{-28.6, -17.875}, {28.6, 17.875}})));
      Connectors.RealInput pwm_rotor_cmd[4];
      // output
      // [m] position: position of drone c.g. in world frame represented in world frame
      // [m/s] velocity: velocity of drone c.g. in world frame represented in body frame
      // [-] quaternion: attitude of drone body with repsect to world frame
      // [rad/s] rate: angular rate of drone body with respect to world frame represented in body frame
      // [m/s^2] acceleration: acceleration of drone c.g. in world frame represented in body frame
      // [m/s^2, rad] acceleration of p(cg) in world frame(w) expressed in body frame(b), euler angle from world(w) to body(b)
      // [rad] euler angle: attitude of drone body with respect to world frame (intrinsic rotation, 3-2-1)
      Connectors.SensorBus sensor annotation(
        Placement(transformation(origin = {120, 7}, extent = {{-19.8, -12.375}, {19.8, 12.375}}), iconTransformation(origin = {129.6, 1.75}, extent = {{-27.6, -17.25}, {27.6, 17.25}})));
      Connectors.RealOutput pos_w_p_w_meas[3], vel_w_p_b_meas[3];
      Connectors.RealOutput quat_wb_meas[4], rate_wb_b_meas[3];
      Connectors.RealOutput acc_w_p_b_meas[3], euler_wb_meas[3];
      // discrete buffers (internal)
      discrete Real pos_w_p_w_buf[3](start={0.0,0.0,0.0}, each fixed=true);
      discrete Real vel_w_p_b_buf[3](start={0.0,0.0,0.0}, each fixed=true);
      discrete Real acc_w_p_b_buf[3](start={0.0,0.0,0.0}, each fixed=true);
      discrete Real quat_wb_buf[4](start={1.0,0.0,0.0,0.0}, each fixed=true);
      discrete Real euler_wb_buf[3](start={0.0,0.0,0.0}, each fixed=true);
      discrete Real rate_wb_b_buf[3](start={0.0,0.0,0.0}, each fixed=true);
      
    algorithm
// pwm sampling of ESC/servo
      when sample(0, actuator_sample_period) then
        if fidelity == 1 then
          for idx in 1:4 loop
            quad_low.omega_rotor_cmd[idx] := (quad_low.omega_rotor_max - quad_low.omega_rotor_min)*(clip(pwm_rotor_cmd[idx], pwm_min, pwm_max) - pwm_min)/(pwm_max - pwm_min) + quad_low.omega_rotor_min;
          end for;
        end if;
      end when;
      when sample(0, sensor_sample_period) then
        if fidelity == 1 then
          pos_w_p_w_buf := quad_low.position_w_p_w;
          vel_w_p_b_buf := quad_low.velocity_w_p_b;
          acc_w_p_b_buf := {0.0,0.0,0.0};
          quat_wb_buf := quad_low.quaternion_wb;
          euler_wb_buf := quat2eul(quad_low.quaternion_wb);
          rate_wb_b_buf := quad_low.rate_wb_b+gyroatk.omega_false;
        end if;
      end when;
          
      equation 
      connect(control.pwm_1, pwm_rotor_cmd[1]);
      connect(control.pwm_2, pwm_rotor_cmd[2]);
      connect(control.pwm_3, pwm_rotor_cmd[3]);
      connect(control.pwm_4, pwm_rotor_cmd[4]);
      connect(sensor.x_w_p_w, pos_w_p_w_meas[1]);
      connect(sensor.y_w_p_w, pos_w_p_w_meas[2]);
      connect(sensor.z_w_p_w, pos_w_p_w_meas[3]);
      connect(sensor.u_w_p_b, vel_w_p_b_meas[1]);
      connect(sensor.v_w_p_b, vel_w_p_b_meas[2]);
      connect(sensor.w_w_p_b, vel_w_p_b_meas[3]);
      connect(sensor.ax_w_p_b_meas, acc_w_p_b_meas[1]);
      connect(sensor.ay_w_p_b_meas, acc_w_p_b_meas[2]);
      connect(sensor.az_w_p_b_meas, acc_w_p_b_meas[3]);
      connect(sensor.q0_wb, quat_wb_meas[1]);
      connect(sensor.q1_wb, quat_wb_meas[2]);
      connect(sensor.q2_wb, quat_wb_meas[3]);
      connect(sensor.q3_wb, quat_wb_meas[4]);
      connect(sensor.phi_wb, euler_wb_meas[1]);
      connect(sensor.theta_wb, euler_wb_meas[2]);
      connect(sensor.psi_wb, euler_wb_meas[3]);
      connect(sensor.p_wb_b, rate_wb_b_meas[1]);
      connect(sensor.q_wb_b, rate_wb_b_meas[2]);
      connect(sensor.r_wb_b, rate_wb_b_meas[3]);
// equation for sensor sampling
      pos_w_p_w_meas = pre(pos_w_p_w_buf);
      vel_w_p_b_meas = pre(vel_w_p_b_buf);
      acc_w_p_b_meas = pre(acc_w_p_b_buf);
      quat_wb_meas = pre(quat_wb_buf);
      euler_wb_meas[1] = pre(euler_wb_buf[1]);
      euler_wb_meas[2] = pre(euler_wb_buf[2]);
      euler_wb_meas[3] = mod(pre(euler_wb_buf[3])+Constants.pi,2*Constants.pi)-Constants.pi;
      rate_wb_b_meas = pre(rate_wb_b_buf);
    annotation(
        Icon(graphics = {Ellipse(origin = {-70, 70}, extent = {{-30, 30}, {30, -30}}), Ellipse(origin = {70, 70}, extent = {{-30, 30}, {30, -30}}), Ellipse(origin = {-70, -70}, extent = {{-30, 30}, {30, -30}}), Ellipse(origin = {70, -70}, extent = {{-30, 30}, {30, -30}}), Polygon(points = {{-20, 40}, {-40, 20}, {-40, -20}, {-20, -40}, {20, -40}, {40, -20}, {40, 20}, {20, 40}, {0, 40}, {-20, 40}}), Ellipse(origin = {-70, 70}, fillPattern = FillPattern.Solid, extent = {{-2, 2}, {2, -2}}), Polygon(origin = {-48, 48}, points = {{-22, 22}, {14, -22}, {20, -16}, {22, -14}, {-22, 22}, {-22, 22}}), Polygon(origin = {48, 48}, points = {{22, 22}, {-14, -22}, {-22, -14}, {22, 22}, {22, 22}}), Ellipse(origin = {70, -70}, fillPattern = FillPattern.Solid, extent = {{-2, 2}, {2, -2}}), Ellipse(origin = {70, 70}, fillPattern = FillPattern.Solid, extent = {{-2, 2}, {2, -2}}), Polygon(origin = {48, -48}, points = {{22, -22}, {-14, 22}, {-22, 14}, {22, -22}, {22, -22}}), Ellipse(origin = {-70, -70}, fillPattern = FillPattern.Solid, extent = {{-2, 2}, {2, -2}}), Polygon(origin = {-48, -48}, points = {{-22, -22}, {22, 14}, {14, 22}, {-22, -22}, {-22, -22}}), Ellipse(origin = {70, 54}, extent = {{-2, -14}, {2, 14}}), Ellipse(origin = {70, 86}, extent = {{-2, -14}, {2, 14}}), Ellipse(origin = {-70, 86}, extent = {{-2, -14}, {2, 14}}), Ellipse(origin = {-70, 54}, extent = {{-2, -14}, {2, 14}}), Ellipse(origin = {-70, -54}, extent = {{-2, -14}, {2, 14}}), Ellipse(origin = {-70, -86}, extent = {{-2, -14}, {2, 14}}), Ellipse(origin = {70, -54}, extent = {{-2, -14}, {2, 14}}), Ellipse(origin = {70, -86}, extent = {{-2, -14}, {2, 14}}), Text(extent = {{-40, 18}, {40, -18}}, textString = "Quadrotor")}));
    end Quadrotor;

    model Controller
      // multi-fidelity controller model
      // load packages
      import GSQuad.Constants;
      // parameters
      // [sec] operational frequency
      parameter Real update_period = 0.005;
      // [-] minimum/maximum PWM
      parameter Real pwm_min = 1000;
      parameter Real pwm_max = 2000;
      // setup controller type by chaning fidelity and load different controller
      parameter Integer fidelity = 1;
      EulerPID euler_pid(update_interval = update_period);
      //QuaternionPID quat_pid(update_interval = update_period);
      // input: sensor feedback from drone (refer to definitions in quadrotor model)
      Connectors.SensorBus sensor annotation(
        Placement(transformation(origin = {120.2, 6.375}, extent = {{-20.2, -12.625}, {20.2, 12.625}}), iconTransformation(origin = {-131, -61.125}, extent = {{-29, -18.125}, {29, 18.25}})));
      Connectors.RealInput pos_w_p_w_fdbk[3], vel_w_p_b_fdbk[3], acc_w_p_b_fdbk[3], quat_wb_fdbk[4], euler_wb_fdbk[3], rate_wb_b_fdbk[3];
      // input: rc joystick input (normalized, roll/pitch/yaw/throttle)
      Connectors.RCBus rc annotation(
        Placement(transformation(origin = {-120, 39}, extent = {{-19.8, -12.375}, {19.8, 12.375}}), iconTransformation(origin = {-128.4, 41.75}, extent = {{-27.6, -17.25}, {27.6, 17.25}})));
      Connectors.RealInput rc_input[4];
      // input: mavlink input (waypoint in world frame)
      Connectors.MAVLinkBus mavlink annotation(
        Placement(transformation(origin = {-120, -13}, extent = {{-19.8, -12.375}, {19.8, 12.375}}), iconTransformation(origin = {-128.4, -8.25}, extent = {{-27.6, -17.25}, {27.6, 17.25}})));
      Connectors.RealInput position_setpoint[3], yaw_setpoint;
      // output, control singal (pwm)
      Connectors.ControlBus control annotation(
        Placement(transformation(origin = {-120.2, -61.625}, extent = {{-20.2, -12.625}, {20.2, 12.625}}), iconTransformation(origin = {131, 2.875}, extent = {{-29, -18.125}, {29, 18.125}})));
      Connectors.RealOutput pwm_rotor_cmd[4];
      // discrete buffers (internal)
      //Real pwm_rotor_cmd_buf[4](start={pwm_min,pwm_min,pwm_min,pwm_min}, each fixed=true);
      
    equation
      connect(sensor.x_w_p_w, pos_w_p_w_fdbk[1]);
      connect(sensor.y_w_p_w, pos_w_p_w_fdbk[2]);
      connect(sensor.z_w_p_w, pos_w_p_w_fdbk[3]);
      connect(sensor.u_w_p_b, vel_w_p_b_fdbk[1]);
      connect(sensor.v_w_p_b, vel_w_p_b_fdbk[2]);
      connect(sensor.w_w_p_b, vel_w_p_b_fdbk[3]);
      connect(sensor.ax_w_p_b_meas, acc_w_p_b_fdbk[1]);
      connect(sensor.ay_w_p_b_meas, acc_w_p_b_fdbk[2]);
      connect(sensor.az_w_p_b_meas, acc_w_p_b_fdbk[3]);
      connect(sensor.q0_wb, quat_wb_fdbk[1]);
      connect(sensor.q1_wb, quat_wb_fdbk[2]);
      connect(sensor.q2_wb, quat_wb_fdbk[3]);
      connect(sensor.q3_wb, quat_wb_fdbk[4]);
      connect(sensor.phi_wb, euler_wb_fdbk[1]);
      connect(sensor.theta_wb, euler_wb_fdbk[2]);
      connect(sensor.psi_wb, euler_wb_fdbk[3]);
      connect(sensor.p_wb_b, rate_wb_b_fdbk[1]);
      connect(sensor.q_wb_b, rate_wb_b_fdbk[2]);
      connect(sensor.r_wb_b, rate_wb_b_fdbk[3]);
      connect(rc.roll, rc_input[1]);
      connect(rc.pitch, rc_input[2]);
      connect(rc.yaw, rc_input[3]);
      connect(rc.throttle, rc_input[4]);
      connect(mavlink.x_w, position_setpoint[1]);
      connect(mavlink.y_w, position_setpoint[2]);
      connect(mavlink.z_w, position_setpoint[3]);
      connect(mavlink.yaw_w, yaw_setpoint);
      connect(control.pwm_1, pwm_rotor_cmd[1]);
      connect(control.pwm_2, pwm_rotor_cmd[2]);
      connect(control.pwm_3, pwm_rotor_cmd[3]);
      connect(control.pwm_4, pwm_rotor_cmd[4]);
      //pwm_rotor_cmd = pre(pwm_rotor_cmd_buf);
    
      for i in 1:4 loop
        pwm_rotor_cmd[i] = pwm_min+(pwm_max-pwm_min)*pre(euler_pid.normalized_ctrl_input[i]);
      end for;
    
    algorithm
// algorithm models pwm sampling of ESC/servo
      when sample(0, update_period) then
        if fidelity == 1 then
          euler_pid.position_setpoint := position_setpoint;
          euler_pid.yaw_setpoint := yaw_setpoint;
          euler_pid.pos_w_p_w_fdbk := pos_w_p_w_fdbk;
          euler_pid.vel_w_p_b_fdbk := vel_w_p_b_fdbk;
          euler_pid.euler_wb_fdbk := euler_wb_fdbk;
          euler_pid.rate_wb_b_fdbk := rate_wb_b_fdbk;
          
        //elseif fidelity == 2 then
        //  quat_pid.position_setpoint := position_setpoint;
        //  quat_pid.yaw_setpoint := yaw_setpoint;
        //  quat_pid.pos_w_p_w_fdbk := pos_w_p_w_fdbk;
        //  quat_pid.vel_w_p_b_fdbk := vel_w_p_b_fdbk;
        //  quat_pid.euler_wb_fdbk := euler_wb_fdbk;
        //  quat_pid.rate_wb_b_fdbk := rate_wb_b_fdbk;
        
        end if;
        
        //for i in 1:4 loop
        //  pwm_rotor_cmd_buf[i] := pwm_min+(pwm_max-pwm_min)*euler_pid.normalized_ctrl_input[i];
        //end for;
        
      end when;
      
      //when sample(0, update_period) then
      //  pwm_rotor_cmd_buf := fill(pwm_min,4)+(pwm_max-pwm_min).*euler_pid.normalized_ctrl_input;
      // end when;
      
      //when sample(0, update_period) then
    
      //end when;
      
      annotation(
        Diagram,
        Icon(graphics = {Rectangle(origin = {0, -14}, fillPattern = FillPattern.Solid, extent = {{-20, 30}, {20, -30}}), Rectangle(origin = {25, 11}, fillPattern = FillPattern.Solid, extent = {{-5, 1}, {5, -1}}), Rectangle(origin = {25, -5}, fillPattern = FillPattern.Solid, extent = {{-5, 1}, {5, -1}}), Rectangle(origin = {25, 3}, fillPattern = FillPattern.Solid, extent = {{-5, 1}, {5, -1}}), Rectangle(origin = {25, -13}, fillPattern = FillPattern.Solid, extent = {{-5, 1}, {5, -1}}), Rectangle(origin = {25, -21}, fillPattern = FillPattern.Solid, extent = {{-5, 1}, {5, -1}}), Rectangle(origin = {-25, 11}, fillPattern = FillPattern.Solid, extent = {{-5, 1}, {5, -1}}), Rectangle(origin = {-25, 3}, fillPattern = FillPattern.Solid, extent = {{-5, 1}, {5, -1}}), Rectangle(origin = {-25, -5}, fillPattern = FillPattern.Solid, extent = {{-5, 1}, {5, -1}}), Rectangle(origin = {-25, -13}, fillPattern = FillPattern.Solid, extent = {{-5, 1}, {5, -1}}), Rectangle(origin = {-25, -21}, fillPattern = FillPattern.Solid, extent = {{-5, 1}, {5, -1}}), Rectangle(lineThickness = 0.75, extent = {{-60, 80}, {60, -80}}), Rectangle(origin = {28, -59}, lineThickness = 1.25, extent = {{-20, 5}, {20, -5}}), Rectangle(origin = {-44, 61}, lineThickness = 0.5, extent = {{-10, 5}, {10, -5}}), Rectangle(origin = {28, -59}, fillPattern = FillPattern.Solid, extent = {{-10, 1}, {10, -1}}), Rectangle(origin = {-28, -59}, lineThickness = 1.25, extent = {{-20, 5}, {20, -5}}), Rectangle(origin = {-28, -59}, fillPattern = FillPattern.Solid, extent = {{-10, 1}, {10, -1}}), Rectangle(origin = {-44, 58}, lineThickness = 0.5, extent = {{-6, 2}, {6, -2}}), Rectangle(origin = {-52, 57}, fillPattern = FillPattern.Solid, extent = {{-2, 1}, {2, -1}}), Rectangle(origin = {-36, 57}, fillPattern = FillPattern.Solid, extent = {{-2, 1}, {2, -1}}), Rectangle(origin = {-51, 63}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-47, 63}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-41, 63}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-37, 63}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-44, 45}, lineThickness = 0.5, extent = {{-10, 5}, {10, -5}}), Rectangle(origin = {-44, 42}, lineThickness = 0.5, extent = {{-6, 2}, {6, -2}}), Rectangle(origin = {-52, 41}, fillPattern = FillPattern.Solid, extent = {{-2, 1}, {2, -1}}), Rectangle(origin = {-36, 41}, fillPattern = FillPattern.Solid, extent = {{-2, 1}, {2, -1}}), Rectangle(origin = {-51, 47}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-47, 47}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-41, 47}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-37, 47}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {16, 61}, lineThickness = 0.5, extent = {{-10, 5}, {10, -5}}), Rectangle(origin = {16, 58}, lineThickness = 0.5, extent = {{-6, 2}, {6, -2}}), Rectangle(origin = {8, 57}, fillPattern = FillPattern.Solid, extent = {{-2, 1}, {2, -1}}), Rectangle(origin = {24, 57}, fillPattern = FillPattern.Solid, extent = {{-2, 1}, {2, -1}}), Rectangle(origin = {9, 63}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {13, 63}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {19, 63}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {23, 63}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-16, 61}, lineThickness = 0.5, extent = {{-10, 5}, {10, -5}}), Rectangle(origin = {-16, 58}, lineThickness = 0.5, extent = {{-6, 2}, {6, -2}}), Rectangle(origin = {-24, 57}, fillPattern = FillPattern.Solid, extent = {{-2, 1}, {2, -1}}), Rectangle(origin = {-8, 57}, fillPattern = FillPattern.Solid, extent = {{-2, 1}, {2, -1}}), Rectangle(origin = {-23, 63}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-19, 63}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-13, 63}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-9, 63}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {44, 45}, lineThickness = 0.5, extent = {{-10, 5}, {10, -5}}), Rectangle(origin = {44, 42}, lineThickness = 0.5, extent = {{-6, 2}, {6, -2}}), Rectangle(origin = {36, 41}, fillPattern = FillPattern.Solid, extent = {{-2, 1}, {2, -1}}), Rectangle(origin = {52, 41}, fillPattern = FillPattern.Solid, extent = {{-2, 1}, {2, -1}}), Rectangle(origin = {37, 47}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {41, 47}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {47, 47}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {51, 47}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {44, 61}, lineThickness = 0.5, extent = {{-10, 5}, {10, -5}}), Rectangle(origin = {44, 58}, lineThickness = 0.5, extent = {{-6, 2}, {6, -2}}), Rectangle(origin = {36, 57}, fillPattern = FillPattern.Solid, extent = {{-2, 1}, {2, -1}}), Rectangle(origin = {52, 57}, fillPattern = FillPattern.Solid, extent = {{-2, 1}, {2, -1}}), Rectangle(origin = {37, 63}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {41, 63}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {47, 63}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {51, 63}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-13, 44}, lineThickness = 0.5, extent = {{-7, 4}, {7, -4}}), Rectangle(origin = {-17, 45}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-9, 45}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-13, 45}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {13, 44}, lineThickness = 0.5, extent = {{-7, 4}, {7, -4}}), Rectangle(origin = {9, 45}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {17, 45}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {13, 45}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Ellipse(origin = {-55, 75}, lineThickness = 0.5, extent = {{-3, 3}, {3, -3}}), Line(origin = {-55, 75}, points = {{-1, 1}, {1, -1}, {1, -1}}), Rectangle(origin = {-25, 29}, lineThickness = 0.5, extent = {{-15, 5}, {15, -5}}), Rectangle(origin = {-25, 26}, lineThickness = 0.5, extent = {{-11, 2}, {11, -2}}), Rectangle(origin = {-37, 31}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-33, 31}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-29, 31}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-25, 31}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-21, 31}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-17, 31}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {-13, 31}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {25, 29}, lineThickness = 0.5, extent = {{-15, 5}, {15, -5}}), Rectangle(origin = {25, 26}, lineThickness = 0.5, extent = {{-11, 2}, {11, -2}}), Rectangle(origin = {13, 31}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {17, 31}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {21, 31}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {25, 31}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {29, 31}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {33, 31}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Rectangle(origin = {37, 31}, lineThickness = 0.5, extent = {{-1, 1}, {1, -1}}), Line(origin = {-55, 75}, points = {{1, 1}, {-1, -1}, {-1, -1}}), Ellipse(origin = {-55, 75}, lineThickness = 0.5, extent = {{-3, 3}, {3, -3}}), Line(origin = {-55, 75}, points = {{-1, 1}, {1, -1}, {1, -1}}), Line(origin = {-55, 75}, points = {{1, 1}, {-1, -1}, {-1, -1}}), Ellipse(origin = {55, 75}, lineThickness = 0.5, extent = {{-3, 3}, {3, -3}}), Line(origin = {55, 75}, points = {{-1, 1}, {1, -1}, {1, -1}}), Line(origin = {55, 75}, points = {{1, 1}, {-1, -1}, {-1, -1}}), Ellipse(origin = {55, 75}, lineThickness = 0.5, extent = {{-3, 3}, {3, -3}}), Line(origin = {55, 75}, points = {{-1, 1}, {1, -1}, {1, -1}}), Line(origin = {55, 75}, points = {{1, 1}, {-1, -1}, {-1, -1}}), Ellipse(origin = {-55, -75}, lineThickness = 0.5, extent = {{-3, 3}, {3, -3}}), Line(origin = {-55, -75}, points = {{-1, 1}, {1, -1}, {1, -1}}), Line(origin = {-55, -75}, points = {{1, 1}, {-1, -1}, {-1, -1}}), Ellipse(origin = {-55, -75}, lineThickness = 0.5, extent = {{-3, 3}, {3, -3}}), Line(origin = {-55, -75}, points = {{-1, 1}, {1, -1}, {1, -1}}), Line(origin = {-55, -75}, points = {{1, 1}, {-1, -1}, {-1, -1}}), Ellipse(origin = {55, -75}, lineThickness = 0.5, extent = {{-3, 3}, {3, -3}}), Line(origin = {55, -75}, points = {{-1, 1}, {1, -1}, {1, -1}}), Line(origin = {55, -75}, points = {{1, 1}, {-1, -1}, {-1, -1}}), Ellipse(origin = {55, -75}, lineThickness = 0.5, extent = {{-3, 3}, {3, -3}}), Line(origin = {55, -75}, points = {{-1, 1}, {1, -1}, {1, -1}}), Line(origin = {55, -75}, points = {{1, 1}, {-1, -1}, {-1, -1}}), Rectangle(origin = {-25, -29}, fillPattern = FillPattern.Solid, extent = {{-5, 1}, {5, -1}}), Rectangle(origin = {-25, -37}, fillPattern = FillPattern.Solid, extent = {{-5, 1}, {5, -1}}), Rectangle(origin = {25, -29}, fillPattern = FillPattern.Solid, extent = {{-5, 1}, {5, -1}}), Rectangle(origin = {25, -37}, fillPattern = FillPattern.Solid, extent = {{-5, 1}, {5, -1}}), Text(origin = {0, -90}, extent = {{-60, 10}, {60, -10}}, textString = "Controller")}));
    end Controller;

    model QuadLowFidelity
      // note: low-fidelity quadrotor model, considering quadrotor dynamics only (w/o aerodynamics, motor, sensor dynamics)
      // load packages
      import GSQuad.Constants;
      import GSQuad.Utils.clip;
      import GSQuad.Utils.quat2rot;
      import GSQuad.Utils.hatmap;
      import GSQuad.Utils.quatprod;
      import GSQuad.Utils.cross3;
      import GSQuad.Utils.inv3x3;
      // parameters
      // [kg] mass of quadrotor
      parameter Real mass = 0.942 + 4*0.15 + 1.072 + 4*0.037;
      // [kg*m^2] x moment of inertia (computed by hand using parallel axis theorem 0.0233)
      parameter Real Ixx = (1/12*(0.942 + 1.072)*(0.11^2 + 0.11^2)) + 4*(1/12*0.15*0.328^2*sin(45/180*Constants.pi)^2 + 0.15*(0.164^2*sin(45/180*Constants.pi)^2 + 0.025^2)) + 4*(1/12*0.037*(0.01^2 + 3*0.02^2) + 0.037*(0.328^2*sin(45/180*Constants.pi)^2 + 0.025^2));
      // [kg*m^2] y moment of inertia (computed by hand using parallel axis theorem 0.0344)
      parameter Real Iyy = (1/12*(0.942 + 1.072)*(0.11^2 + 0.28^2)) + 4*(1/12*0.15*0.328^2*sin(45/180*Constants.pi)^2 + 0.15*(0.164^2*sin(45/180*Constants.pi)^2 + 0.025^2)) + 4*(1/12*0.037*(0.01^2 + 3*0.02^2) + 0.037*(0.328^2*sin(45/180*Constants.pi)^2 + 0.025^2));
      // [kg*m^2] z moment of inertia (computed by hand using parallel axis theorem 0.0527)
      parameter Real Izz = (1/12*(0.942 + 1.072)*(0.11^2 + 0.28^2)) + 4*(1/3*0.15*0.328^2) + 4*(1/2*0.037*0.02^2 + 0.037*0.328^2);
      // [kg*m^2] xy product inertia
      parameter Real Ixy = 0.0;
      // [kg*m^2] yz product inertia
      parameter Real Iyz = 0.0;
      // [kg*m^2] xz product inertia
      parameter Real Ixz = 0.0;
      // [kg*m^2] rotor inertia
      parameter Real Ir = 0.0;
      // get inertia matrix and its inversion
      parameter Real J[3,3] = [Ixx, Ixy, Ixz; Ixy, Iyy, Iyz; Ixz, Iyz, Izz];
      parameter Real Jinv[3,3] = inv3x3([Ixx, Ixy, Ixz; Ixy, Iyy, Iyz; Ixz, Iyz, Izz]);
      // [m] arm length from cg to each rotor
      parameter Real d_arm = 0.328;
      // [m] rotor position w.r.t cg
      parameter Real rotor_pos[3, 4] = d_arm*[cos(45/180*Constants.pi), cos(135/180*Constants.pi), cos(225/180*Constants.pi), cos(315/180*Constants.pi); sin(45/180*Constants.pi), sin(135/180*Constants.pi), sin(225/180*Constants.pi), sin(315/180*Constants.pi); 0.0, 0.0, 0.0, 0.0];
      // [-] rotating direction of rotors (+1: CCW, -1: CW, CCW rotation = CW torque = +Z yaw moment)
      parameter Integer rotor_dir[4] = {1, -1, 1, -1};
      // [N/(rad/s)^2] thrust coefficient
      parameter Real k_eta = 1.7*5.570e-6;
      // [N*m/(rad/s)^2] yaw moment coefficient
      parameter Real k_m = 1.7*0.136e-6;
      // [rad/s] rotor maximum rotational speed
      parameter Real omega_rotor_max = 1500;
      // [rad/s] rotor minimum rotational speed
      parameter Real omega_rotor_min = 0;
      // [-] feedback correction term for quaternion (numerical constraints as lagrange multiplier)
      parameter Real quat_fdbk_correction = 0.1;
      // inputs
      // [rad/s] rotational speed command from ESC
      discrete Real omega_rotor_cmd[4];
      // states
      // [m] position of p(cg) in world frame(w) expressed in world frame(w) = ned position from origin
      Real position_w_p_w[3](start = {0, 0, 0}, each fixed = true);
      // [m/s] velocity of p(cg) in world frame(w) expressed in body frame(b) = body velocity
      Real velocity_w_p_b[3](start = {0, 0, 0}, each fixed = true);
      // [-] attitude of body frame(b) relative to world frame(w) = attitude w.r.t ned (intrinsic)
      Real quaternion_wb[4](start = {1, 0, 0, 0}, each fixed = true);
      // [rad/s] rate of body frame(b) relative to world frame(w) expressed in body frame(b) = p,q,r
      Real rate_wb_b[3](start = {0, 0, 0}, each fixed = true);
      // [rad/s] rotational speed of rotors
      Real omega_motor[4](start = {0, 0, 0, 0}, each fixed = true);
      // internal states
      // [N, N*m] force/moment in body coordinates
      Real F_b[3], M_b[3];
      // [-] rotation matrix
      Real C_wb[3, 3];
    equation
// compute body forces and moments
      for idx in 1:4 loop
        omega_motor[idx] = clip(pre(omega_rotor_cmd[idx]), omega_rotor_min, omega_rotor_max);
      end for;
      F_b = {0.0, 0.0, -k_eta*sum(omega_motor[idx]^2 for idx in 1:4)};
      M_b = sum(cross(rotor_pos[:, idx], {0, 0, -k_eta*omega_motor[idx]^2}) + (rotor_dir[idx]*{0, 0, k_m*omega_motor[idx]^2}) for idx in 1:4);
// compute coordinate transformation matrix from w to b
      C_wb = transpose(quat2rot(quaternion_wb));
// compute the derivative
      der(position_w_p_w) = transpose(C_wb)*velocity_w_p_b;
      der(velocity_w_p_b) = -cross3(rate_wb_b, velocity_w_p_b) + F_b./mass + C_wb*{0, 0, Constants.g};
      der(quaternion_wb) = 0.5*quatprod({0.0, rate_wb_b[1], rate_wb_b[2], rate_wb_b[3]}, quaternion_wb) + quat_fdbk_correction*(1 - sum(quaternion_wb.*quaternion_wb))*quaternion_wb;
      der(rate_wb_b) = Jinv*(-hatmap({rate_wb_b[1], rate_wb_b[2], rate_wb_b[3]})*(J*{rate_wb_b[1], rate_wb_b[2], rate_wb_b[3]}) + M_b);
      annotation(
        Icon(graphics = {Rectangle(origin = {0, 60}, fillColor = {200, 200, 200}, fillPattern = FillPattern.Solid, extent = {{-20, -20}, {20, 20}}), Line(origin = {0, 60}, points = {{0, 0}, {60, 0}}, color = {0, 0, 255}, thickness = 2, arrow = {Arrow.None, Arrow.Filled}), Text(origin = {21, 62}, extent = {{44, -3}, {64, 8}}, textString = "Velocity"), Text(origin = {53, 58}, extent = {{-114, -6}, {-78, 17}}, textString = "No Drag Force"), Rectangle(fillColor = {230, 230, 230}, fillPattern = FillPattern.Solid, extent = {{-20, -20}, {20, 20}}), Text(origin = {0, -24},extent = {{-20, 5}, {20, 15}}, textString = "Motor"), Line(origin = {18.0957, -0.670213},points = {{-80, 0}, {-40, 0}}, color = {0, 0, 255}, thickness = 2, arrow = {Arrow.None, Arrow.Filled}), Text(origin = {18, 9},extent = {{-88, -2}, {-64, 5}}, textString = "ω_{cmd}"), Line(points = {{0, -20}, {0, -60}}, color = {0, 128, 0}, thickness = 2, arrow = {Arrow.None, Arrow.Filled}), Text(origin = {0, -26}, extent = {{3, -25}, {21, -16}}, textString = "ω"), Line(origin = {-20, 0},points = {{40, 0}, {80, 0}}, color = {255, 0, 0}, thickness = 2, arrow = {Arrow.None, Arrow.Filled}), Text(origin = {5, 7},extent = {{56, -3}, {79, 8}}, textString = "Thrust T"), Text(origin = {6, -35}, extent = {{-72, -37}, {72, -30}}, textString = "ω_{cmd}=ω, T = k_{T}ω^{2}"), Text(origin = {0, 36},extent = {{-20, 5}, {20, 15}}, textString = "Chassis")}));
    end QuadLowFidelity;

    model Joystick
      // parameters
      // [sec] output command update rate
      parameter Real sample_period = 0.01;
      // output: normalized rc signal (roll/pitch/yaw [-1, +1], throttle [0, 1])
      Connectors.RealOutput stick_cmd[4];
      Connectors.RCBus rc annotation(
        Placement(transformation(origin = {120.2, 6.375}, extent = {{-20.2, -12.625}, {20.2, 12.625}}), iconTransformation(origin = {129, -17.125}, extent = {{-29, -18.125}, {29, 18.25}})));
      // discrete buffers (internal)
      discrete Real stick_cmd_buf[4](start={0.0,0.0,0.0,0.0}, each fixed=true);
    equation
      connect(rc.roll, stick_cmd[1]);
      connect(rc.pitch, stick_cmd[2]);
      connect(rc.yaw, stick_cmd[3]);
      connect(rc.throttle, stick_cmd[4]);
      stick_cmd = pre(stick_cmd_buf);
    algorithm
      when sample(0, sample_period) then
// default mode: zero commands from joystick
        stick_cmd_buf := {0.0,0.0,0.0,0.0};
      end when;
      annotation(
        Icon(graphics = {Rectangle(extent = {{-80, 60}, {80, -60}}), Ellipse(origin = {-39, 25}, extent = {{-25, 25}, {25, -25}}), Ellipse(origin = {39, 25}, extent = {{-25, 25}, {25, -25}}), Rectangle(origin = {0, -30}, extent = {{-60, 20}, {60, -20}}), Text(origin = {0, -30}, extent = {{-40, 10}, {40, -10}}, textString = "Joystick"), Rectangle(origin = {0, 76}, extent = {{-4, 16}, {4, -16}}), Ellipse(origin = {-39, 25}, extent = {{-3, 3}, {3, -3}}), Rectangle(origin = {-39, 25}, extent = {{-19, 11}, {19, -11}}), Rectangle(origin = {39, 25}, extent = {{-19, 11}, {19, -11}}), Ellipse(origin = {39, 19}, extent = {{-3, 3}, {3, -3}})}));
    end Joystick;

    model MissionPlanner
      // parameters
      // [sec] output command update rate
      parameter Real sample_period = 0.05;
      // output: mavlink signal (simplified: currently support only waypoints)
      // [m] waypoint cooridnate defined in world frame (w), NED coordinates
      Connectors.RealOutput position_setpoint_w[3];
      // [rad] desired yaw attitude at waypoint (w), NED coordinates
      Connectors.RealOutput yaw_setpoint_w;
      Connectors.MAVLinkBus mavlink annotation(
        Placement(transformation(origin = {120.2, 6.375}, extent = {{-20.2, -12.625}, {20.2, 12.625}}), iconTransformation(origin = {129, -17.125}, extent = {{-29, -18.125}, {29, 18.25}})));
      // discrete buffers (internal)
      discrete Real position_setpoint_w_buf[3](start={0.0,0.0,0.0}, each fixed=true);
      discrete Real yaw_setpoint_w_buf(start=0.0, fixed=true);
    equation
      connect(mavlink.x_w, position_setpoint_w[1]);
      connect(mavlink.y_w, position_setpoint_w[2]);
      connect(mavlink.z_w, position_setpoint_w[3]);
      connect(mavlink.yaw_w, yaw_setpoint_w);
      position_setpoint_w = pre(position_setpoint_w_buf);
      yaw_setpoint_w = pre(yaw_setpoint_w_buf);
    algorithm
      when sample(0, sample_period) then
// default waypoint in NED coordinate
        position_setpoint_w_buf := {3.0, 5.0, -7.0};
        yaw_setpoint_w_buf := 0.01;
      end when;
      annotation(
        Icon(graphics = {Rectangle(origin = {0, 35}, extent = {{-60, 35}, {60, -35}}), Polygon(origin = {0, -26}, points = {{-60, 26}, {-80, -26}, {80, -26}, {60, 26}, {-60, 26}}), Rectangle(origin = {0, 35}, extent = {{-56, 31}, {56, -31}}), Text(origin = {0, 36}, extent = {{-56, 18}, {56, -18}}, textString = "Mission Planner"), Polygon(origin = {0, -25}, points = {{-58, 23}, {-76, -23}, {76, -23}, {58, 23}, {-58, 23}, {-58, 23}}), Polygon(origin = {4, -13}, points = {{-60, 9}, {-68, -11}, {60, -11}, {52, 9}, {52, 9}, {-60, 9}}), Polygon(origin = {-32, -37}, points = {{12, 9}, {10, -7}, {54, -7}, {52, 9}, {52, 9}, {12, 9}})}));
    end MissionPlanner;

    model EulerPID
      // note: Euler angle-based PID controller model, position tracking
      // load packages
      import GSQuad.Constants;
      import GSQuad.Utils.clip;
      import GSQuad.Utils.quat2rot;
      import GSQuad.Utils.eul2rot;
      import GSQuad.Utils.hatmap;
      import Modelica.Math.sin;
      import Modelica.Math.cos;
      import Modelica.Math.atan;
      import Modelica.Math.acos;
      import Modelica.Math.Matrices.inv;
      // parameters
      // [rad/s] rotor maximum rotational speed
      parameter Real omega_rotor_max = 1500;
      // [rad/s] rotor minimum rotational speed
      parameter Real omega_rotor_min = 0;
      // [sec] operational frequency
      parameter Real update_interval;
      // [-] gains
      parameter Real Kp_x = 0.20;
      parameter Real Kp_y = 0.20;
      parameter Real Kp_z = 0.40;
      parameter Real Kp_vx = 0.45;
      parameter Real Ki_vx = 0.10;
      parameter Real Kd_vx = 0.00;
      parameter Real Kp_vy = 0.45;
      parameter Real Ki_vy = 0.10;
      parameter Real Kd_vy = 0.00;
      parameter Real Kp_vz = 0.60;
      parameter Real Ki_vz = 0.10;
      parameter Real Kd_vz = 0.00;
      parameter Real Kp_phi = 1.00;
      parameter Real Kp_the = 1.00;
      parameter Real Kp_psi = 0.80;
      parameter Real Kp_p = 1.60;
      parameter Real Ki_p = 0.00;
      parameter Real Kd_p = 0.60;
      parameter Real Kp_q = 1.60;
      parameter Real Ki_q = 0.00;
      parameter Real Kd_q = 0.60;
      parameter Real Kp_r = 2.00;
      parameter Real Ki_r = 0.00;
      parameter Real Kd_r = 0.80;
      // input
      discrete Real pos_w_p_w_fdbk[3], vel_w_p_b_fdbk[3], euler_wb_fdbk[3], rate_wb_b_fdbk[3];
      discrete Real position_setpoint[3], yaw_setpoint;
      // output
      // [-] rotational speed command from ESC
      discrete Real normalized_ctrl_input[4](start={0,0,0,0}, each fixed=false);
      // states
      // [m] position error
      discrete Real pos_error[3];
      // [m/s] velocity target
      discrete Real vel_target[3];
      // [m/s] velocity error
      discrete Real vel_error[3];
      // compute coordinate transformation matrix from w to b
      discrete Real C_wb[3, 3](start = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, each fixed = false);
      // [m] velocity error integral
      discrete Real vel_error_i[3];
      // [m/s] previous iteration velocity error
      discrete Real vel_error_last[3];
      // [m/s^2] velocity error derivative
      discrete Real vel_error_d[3];
      // [m/s^2] acceleration target
      discrete Real acc_target[3];
      discrete Real acc_z_target, acc_fwd_target, acc_rgt_target;
      // [rad] attitude target
      discrete Real roll_target, pitch_target, yaw_target;
      // [rad] attitude_error
      discrete Real att_error[3];
      // [rad/s] rate target
      discrete Real rate_target[3];
      // [rad/s] rate error
      discrete Real rate_error[3];
      // [rad] angular rate error integral
      discrete Real rate_error_i[3];
      // [rad/s] previous iteration rate error
      discrete Real rate_error_last[3];
      // [rad/s^2] angular rate error derivative
      discrete Real rate_error_d[3];
      // [N, N*m] force/moment target in body coordinates
      discrete Real force_target, moment_target[3];
      discrete Real fm_target[4];
      // [-] current body thrust vector direction (-Z direction of inertial coordinates represented in body coordinates)
      discrete Real att_body_thrust_vec[3];
      // [-] lean angle between inertial/body negative z-axis
      discrete Real lean_angle;
      // [-] required thrust of each rotor
      discrete Real thrust_target[4];
      discrete Real omega_spd_sq_target[4];
      // parameters
      // [kg] mass of quadrotor
      parameter Real mass = 0.942 + 4*0.15 + 1.072 + 4*0.037;
      // [kg*m^2] x moment of inertia (computed by hand using parallel axis theorem 0.0233)
      parameter Real Ixx = (1/12*(0.942 + 1.072)*(0.11^2 + 0.11^2)) + 4*(1/12*0.15*0.328^2*sin(45/180*Constants.pi)^2 + 0.15*(0.164^2*sin(45/180*Constants.pi)^2 + 0.025^2)) + 4*(1/12*0.037*(0.01^2 + 3*0.02^2) + 0.037*(0.328^2*sin(45/180*Constants.pi)^2 + 0.025^2));
      // [kg*m^2] y moment of inertia (computed by hand using parallel axis theorem 0.0344)
      parameter Real Iyy = (1/12*(0.942 + 1.072)*(0.11^2 + 0.28^2)) + 4*(1/12*0.15*0.328^2*sin(45/180*Constants.pi)^2 + 0.15*(0.164^2*sin(45/180*Constants.pi)^2 + 0.025^2)) + 4*(1/12*0.037*(0.01^2 + 3*0.02^2) + 0.037*(0.328^2*sin(45/180*Constants.pi)^2 + 0.025^2));
      // [kg*m^2] z moment of inertia (computed by hand using parallel axis theorem 0.0527)
      parameter Real Izz = (1/12*(0.942 + 1.072)*(0.11^2 + 0.28^2)) + 4*(1/3*0.15*0.328^2) + 4*(1/2*0.037*0.02^2 + 0.037*0.328^2);
      // [kg*m^2] xy product inertia
      parameter Real Ixy = 0.0;
      // [kg*m^2] yz product inertia
      parameter Real Iyz = 0.0;
      // [kg*m^2] xz product inertia
      parameter Real Ixz = 0.0;
      // [kg*m^2] inertia
      parameter Real J[3, 3] = [Ixx, Ixy, Ixz; Ixy, Iyy, Iyz; Ixz, Iyz, Izz];
      // [N/(N,N*m)] control effectiveness pseudo-inverse
      //parameter Real CA[4,4] = [1, 1, 1, 1; 0.328*cos(45/180*Constants.pi), 0.328*cos(135/180*Constants.pi), 0.328*cos(225/180*Constants.pi), 0.328*cos(315/180*Constants.pi); 0.328*sin(45/180*Constants.pi), 0.328*sin(135/180*Constants.pi), 0.328*sin(225/180*Constants.pi), 0.328*sin(315/180*Constants.pi); 0.136e-6/5.570e-6, -0.136e-6/5.570e-6, 0.136e-6/5.570e-6, -0.136e-6/5.570e-6];
      parameter Real pinv_CA[4, 4] = {{0.2500, 1.0779, 1.0779, 10.2390}, {0.2500, -1.0779, 1.0779, -10.2390}, {0.2500, -1.0779, -1.0779, 10.2390}, {0.2500, 1.0779, -1.0779, -10.2390}};
      // [N/(rad/s)^2] thrust coefficient
      parameter Real k_eta = 1.7*5.570e-6;
      
    equation
    
    algorithm
      when sample(0, update_interval) then
        pos_error := position_setpoint - pos_w_p_w_fdbk;
        vel_target := {Kp_x, Kp_y, Kp_z}.*pos_error;
        C_wb := transpose(eul2rot(euler_wb_fdbk));
        vel_error := vel_target - transpose(C_wb)*vel_w_p_b_fdbk;
        vel_error_i := vel_error_i + vel_error*update_interval;
        vel_error_d := (vel_error - vel_error_last)/update_interval;
        vel_error_last := vel_error;
        acc_target := {Kp_vx, Kp_vy, Kp_vz}.*vel_error + {Ki_vx, Ki_vy, Ki_vz}.*vel_error_i + {Kd_vx, Kd_vy, Kd_vz}.*vel_error_d - {0.0, 0.0, Constants.g};
        acc_z_target := -acc_target[3];
        acc_fwd_target := acc_target[1]*cos(euler_wb_fdbk[3]) + acc_target[2]*sin(euler_wb_fdbk[3]);
        acc_rgt_target := -acc_target[1]*sin(euler_wb_fdbk[3]) + acc_target[2]*cos(euler_wb_fdbk[3]);
        pitch_target := atan(-acc_fwd_target/Constants.g);
        roll_target := atan(acc_rgt_target*cos(pitch_target)/Constants.g);
        yaw_target := yaw_setpoint;
        att_error := {roll_target, pitch_target, yaw_target} - euler_wb_fdbk;
        rate_target := {Kp_phi, Kp_the, Kp_psi}.*att_error;
        rate_error := rate_target - rate_wb_b_fdbk;
        rate_error_i := rate_error_i + rate_error*update_interval;
        rate_error_d := (rate_error - rate_error_last)/update_interval;
        rate_error_last := rate_error;
        moment_target := J*({Kp_p, Kp_q, Kp_r}.*rate_error + {Ki_p, Ki_q, Ki_r}.*rate_error_i + {Kd_p, Kd_q, Kd_r}.*rate_error_d) + hatmap(rate_wb_b_fdbk)*(J*rate_wb_b_fdbk);
        att_body_thrust_vec := C_wb*{0.0, 0.0, -1.0};
        lean_angle := acos(clip({0.0, 0.0, -1.0}*att_body_thrust_vec, -1, 1));
        force_target := mass*acc_z_target/cos(lean_angle);
        fm_target := cat(1, {force_target}, moment_target);
        
        thrust_target := pinv_CA*fm_target;
        omega_spd_sq_target := thrust_target./k_eta;
        for i in 1:4 loop
          normalized_ctrl_input[i] := (sqrt(omega_spd_sq_target[i])-omega_rotor_min)/(omega_rotor_max-omega_rotor_min)+omega_rotor_min;
        end for;
      end when;
      
    
        
    end EulerPID;

    model QuaternionPID
      // note: Quaternion-based PID controller model, surrogate of Arducopter controller
      
      
      // load packages
      import GSQuad.Constants;
      import GSQuad.Utils.clip;
      import GSQuad.Utils.quat2rot;
      import GSQuad.Utils.eul2rot;
      import GSQuad.Utils.hatmap;
      import Modelica.Math.sin;
      import Modelica.Math.cos;
      import Modelica.Math.atan;
      import Modelica.Math.acos;
      import Modelica.Math.Matrices.inv;
      
      
      // parameters
      // [rad/s] rotor maximum rotational speed
      parameter Real omega_rotor_max = 1500;
      // [rad/s] rotor minimum rotational speed
      parameter Real omega_rotor_min = 0;
      // [sec] operational frequency
      parameter Real update_interval = 0.01;
      // [-] gains
      // outer loop control gains (PSC)
      parameter Real PSC_POSXY_P = 1.0; // xy position P gain (≈ 1/s), higher -> faster XY position correction
      parameter Real PSC_POSZ_P = 1.0; // z position P gain (≈ 1/s), higher -> tighter altitude hold / faster climb-rate command
      parameter Real PSC_VELXY_P = 2.0; // xy velocity P gain (≈ 1/s), XY velocity error -> lateral accel (tilt) command
      parameter Real PSC_VELXY_I = 1.0; // xy velocity I gain (≈ 1/s^2), removes steady XY drift (integrates error to accel cmd)
      parameter Real PSC_VELXY_D = 0.5; // xy velocity D gain (≈ s^0), adds rate-of-change damping on velocity error
      parameter Real PSC_VELZ_P = 5.0; // z velocity P gain (≈ 1/s), climb-rate error -> vertical accel/thrust command
      parameter Real PSC_VELZ_I = 0.0; // z velocity I gain (≈ 1/s^2), remove bias on climb rate
      parameter Real PSC_VELZ_D = 0.0; // z velcotiy D gain (≈ s^0), adds rate-of-change damping on climb-rate error
      // inner loop control gains (ATC)
      parameter Real ATC_ANG_RLL_P = 4.5; // roll angle P gain (≈ 1/s) maps angle error -> desired body rate
      parameter Real ATC_ANG_PIT_P = 4.5; // pitch angle P gain (≈ 1/s)
      parameter Real ATC_ANG_YAW_P = 4.5; // yaw angle P gain (≈ 1/s)
      parameter Real ATC_RAT_RLL_P = 0.135; // roll rate P gain maps body-rate error -> torque/motor mix command
      parameter Real ATC_RAT_RLL_I = 0.135; // roll rate I gain
      parameter Real ATC_RAT_RLL_D = 0.0036; // roll rate D gain
      parameter Real ATC_RAT_PIT_P = 0.135; // pitch rate P gain
      parameter Real ATC_RAT_PIT_I = 0.135; // pitch rate I gain
      parameter Real ATC_RAT_PIT_D = 0.0036; // pitch rate D gain
      parameter Real ATC_RAT_YAW_P = 0.30; // yaw rate P gain
      parameter Real ATC_RAT_YAW_I = 0.02; // yaw rate I gain
      parameter Real ATC_RAT_YAW_D = 0.00; // yaw rate D gain
      // thrust/accel gains
      parameter Real PSC_ACCZ_P = 0.5/10; // z acceleration P gain maps z acceleration error to collective/throttle, sets responsiveness of thrust to accel error
      parameter Real PSC_ACCZ_I = 1.0/10; // z acceleration I gain removes hover bias and improves long-term altitude hold
      parameter Real PSC_ACCZ_D = 0.0/10; // z acceleration D gain usually set to be zero to avoid noise effects on accel feedback
      
      
      // parameters
      parameter Real POSCONTROL_ACCEL_U_MSS = 2.5; // [m/s^2] default vertical acceleration
      parameter Real POSCONTROL_JERK_U_MSSS = 5.0; // [m/s^3] default vertical jerk
      // parameter Real ANGLE_MAX = 3000/100/180*pi;     // [rad] maximum angle command
      // input
      discrete Real pos_w_p_w_fdbk[3], vel_w_p_b_fdbk[3], quat_wb_fdbk[3], rate_wb_b_fdbk[3];
      discrete Real position_setpoint[3], yaw_setpoint;
    
      
      // output
      // [-] rotational speed command from ESC
      discrete Real normalized_ctrl_input[4](start={0,0,0,0}, each fixed=false);
      
      
      // states
      // [m] desired position
      discrete Real pos_desired[3];
      // [m/s] desired velocity
      discrete Real vel_desired[3];
      // [m/s^2] desired acceleration
      discrete Real acc_desired[3];
    
      // [m] position error
    //  discrete Real pos_error[3];
      // [m/s] velocity target
    //  discrete Real vel_target[3];
      // [m/s] velocity error
    //  discrete Real vel_error[3];
      // compute coordinate transformation matrix from w to b
    //  discrete Real C_wb[3, 3](start = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, each fixed = false);
      // [m] velocity error integral
    //  discrete Real vel_error_i[3];
      // [m/s] previous iteration velocity error
    //  discrete Real vel_error_last[3];
      // [m/s^2] velocity error derivative
    //  discrete Real vel_error_d[3];
      // [m/s^2] acceleration target
    //  discrete Real acc_target[3];
    //  discrete Real acc_z_target, acc_fwd_target, acc_rgt_target;
      // [rad] attitude target
    //  discrete Real roll_target, pitch_target, yaw_target;
      // [rad] attitude_error
    //  discrete Real att_error[3];
      // [rad/s] rate target
    //  discrete Real rate_target[3];
      // [rad/s] rate error
    //  discrete Real rate_error[3];
      // [rad] angular rate error integral
    //  discrete Real rate_error_i[3];
      // [rad/s] previous iteration rate error
    //  discrete Real rate_error_last[3];
      // [rad/s^2] angular rate error derivative
    //  discrete Real rate_error_d[3];
      // [N, N*m] force/moment target in body coordinates
    //  discrete Real force_target, moment_target[3];
    //  discrete Real fm_target[4];
      // [-] current body thrust vector direction (-Z direction of inertial coordinates represented in body coordinates)
    //  discrete Real att_body_thrust_vec[3];
      // [-] lean angle between inertial/body negative z-axis
    //  discrete Real lean_angle;
      // [-] required thrust of each rotor
    //  discrete Real thrust_target[4];
    //  discrete Real omega_spd_sq_target[4];
    
    equation
    
    algorithm
      when sample(0, update_interval) then
        pos_desired := pos_desired+vel_desired*update_interval+0.5*acc_desired*update_interval^2;
        vel_desired := vel_desired+acc_desired*update_interval;
      acc_desired := {0.0,0.0,0.0};
      //[sqrt_ctrl(vel_error_x,params.POSCONTROL_JERK_U_MSSS/params.POSCONTROL_JERK_U_MSSS,params.POSCONTROL_JERK_U_MSSS,params.dt_s);
     //                    sqrt_ctrl(vel_error_y,params.POSCONTROL_JERK_U_MSSS/params.POSCONTROL_JERK_U_MSSS,params.POSCONTROL_JERK_U_MSSS,params.dt_s);
     //                    sqrt_ctrl(vel_error_z,params.POSCONTROL_JERK_U_MSSS/params.POSCONTROL_JERK_U_MSSS,params.POSCONTROL_JERK_U_MSSS,params.dt_s)];
        
        
    //    pos_error := position_setpoint - pos_w_p_w_fdbk;
    //    vel_target := {Kp_x, Kp_y, Kp_z}.*pos_error;
    //    C_wb := transpose(eul2rot(euler_wb_fdbk));
    //    vel_error := vel_target - transpose(C_wb)*vel_w_p_b_fdbk;
    //    vel_error_i := vel_error_i + vel_error*update_interval;
    //    vel_error_d := (vel_error - vel_error_last)/update_interval;
    //    vel_error_last := vel_error;
    //    acc_target := {Kp_vx, Kp_vy, Kp_vz}.*vel_error + {Ki_vx, Ki_vy, Ki_vz}.*vel_error_i + {Kd_vx, Kd_vy, Kd_vz}.*vel_error_d - {0.0, 0.0, Constants.g};
    //    acc_z_target := -acc_target[3];
    //    acc_fwd_target := acc_target[1]*cos(euler_wb_fdbk[3]) + acc_target[2]*sin(euler_wb_fdbk[3]);
    //    acc_rgt_target := -acc_target[1]*sin(euler_wb_fdbk[3]) + acc_target[2]*cos(euler_wb_fdbk[3]);
    //    pitch_target := atan(-acc_fwd_target/Constants.g);
    //    roll_target := atan(acc_rgt_target*cos(pitch_target)/Constants.g);
    //    yaw_target := yaw_setpoint;
    //    att_error := {roll_target, pitch_target, yaw_target} - euler_wb_fdbk;
    //    rate_target := {Kp_phi, Kp_the, Kp_psi}.*att_error;
    //    rate_error := rate_target - rate_wb_b_fdbk;
    //    rate_error_i := rate_error_i + rate_error*update_interval;
    //    rate_error_d := (rate_error - rate_error_last)/update_interval;
    //    rate_error_last := rate_error;
    //    moment_target := J*({Kp_p, Kp_q, Kp_r}.*rate_error + {Ki_p, Ki_q, Ki_r}.*rate_error_i + {Kd_p, Kd_q, Kd_r}.*rate_error_d) + hatmap(rate_wb_b_fdbk)*(J*rate_wb_b_fdbk);
    //    att_body_thrust_vec := C_wb*{0.0, 0.0, -1.0};
    //    lean_angle := acos(clip({0.0, 0.0, -1.0}*att_body_thrust_vec, -1, 1));
    //    force_target := mass*acc_z_target/cos(lean_angle);
    //    fm_target := {force_target, moment_target[1], moment_target[2], moment_target[3]};
    //    thrust_target := pinv_CA*fm_target;
    //    omega_spd_sq_target := thrust_target./k_eta;
    //    normalized_ctrl_input := (sqrt(omega_spd_sq_target)-fill(omega_rotor_min,4))./(omega_rotor_max-omega_rotor_min)+fill(omega_rotor_min,4);
        normalized_ctrl_input := {0,0,0,0};
      end when;

    end QuaternionPID;

    model GyroAcousticAtk
    // load packages
      import GSQuad.Utils.clip;
      import GSQuad.Utils.avoidzero;
      import GSQuad.Utils.eul2rot;
      import Modelica.Math.sin;
      import Modelica.Math.cos;
      import Modelica.Math.atan;
      import Modelica.Math.log10;
    // system parameter
      parameter Real sample_interval = 0.1;                                     // [sec] update rate for attack
      // targeted acoustic attack model and parameters
      parameter Real m_d = 2.5e-9;                              // [kg] gyroscope driving mass
      parameter Real m_s = 4.1e-9;                              // [kg] gyroscope sensing mass
      parameter Real w_d = 15e+3*2*Constants.pi;                // [rad/s] gyroscope driving frequency
      parameter Real w_s = 23e+3*2*Constants.pi;                // [rad/s] gyroscope sensing natural frequency
      parameter Real k_d = m_d*w_d^2;                           // [N/m] gyroscope driving spring coefficient
      parameter Real k_s = m_s*w_s^2;                           // [N/m] gyroscope sensing spring coefficient
      parameter Real zeta_d = 1/90;                             // [N/m] gyroscope driving damping coefficient (Q-factor: Qd = 45)
      parameter Real zeta_s = 1/36;                             // [N/m] gyroscope sensing damping coefficient (Q-factor: Qs = 18)
      parameter Real dis_d = 7e-6;                              // [m] driving mass displacement
      parameter Real F_d = k_d*dis_d/(1/2/zeta_d);              // [N] driving force needed, small because of very low damping
      parameter Real W = 0;                                     // [W] power of speaker
      parameter Real dist = 0.01;                               // [m] distance to speaker
      parameter Real p0 = 20*10^(-6);                           // [pa] reference pressure
      parameter Real SPL = 10*log10(avoidzero(W/4/Constants.pi/dist^2/(1.21)/(343)/p0^2)); // [pa] sound pressure level
      parameter Real A = p0*10^(SPL/20)*(2*dis_d)^2;            // [N] acoustic force acting on gyro, suppose area = (2*driving displacement)*(2*driving displacement)
      parameter Real psi_ac = 80.0*Constants.d2r;               // [rad] speaker direction
      parameter Real A_x = A*cos(psi_ac);                       // [N] acoustic force on sensing axis, reference - suggested value 4.0e-9
      parameter Real A_y = A*sin(psi_ac);                       // [N] acoustic force on driving axis, reference - suggested value 16.0e-9
      parameter Real w_ac = 15.0002e+3*2*Constants.PI;          // [rad/s] acoustic attack frequency
      parameter Real epsilon = 0.0*Constants.d2r;               // [rad] misalignment of gyroscope, reference - 1deg
      parameter Real phi_0 = 30*Constants.d2r;                  // [rad] phase shift for acoustic noise compared to driving signal
      parameter Real l_g = 1.0e-6;                              // [m] unit length scale
      parameter Real k_bar = k_d/k_s;
      parameter Real w_1 = w_ac/w_d;
      parameter Real w_2 = w_s/w_d;
      parameter Real w_3 = w_ac/w_s;
      parameter Real D_s = F_d/(m_s*w_d^2*l_g);
      parameter Real D_d = F_d/(m_d*w_d^2*l_g);
      parameter Real X_acx = (A_x/(m_s*w_d^2*l_g))/(w_2^2*sqrt((avoidzero(1-w_3^2))^2+(2*zeta_s*w_3)^2));
      parameter Real X_acy = k_bar*sin(epsilon)*(A_y/(m_d*w_d^2*l_g))/sqrt((avoidzero(1-w_3^2))^2+(2*zeta_s*w_3)^2)*sqrt(1+(2*zeta_d*w_1)^2)/sqrt((avoidzero(cos(epsilon)-w_1^2))^2+(2*zeta_d*w_1*cos(epsilon))^2);
      parameter Real X_d1 = sin(epsilon)*D_s/sqrt((avoidzero(1-w_2^2))^2+(2*zeta_s*w_2)^2);
      parameter Real X_d2 = sin(epsilon)*cos(epsilon)*D_s/sqrt((avoidzero(1-w_2^2))^2+(2*zeta_s*w_2)^2)*sqrt(1+(2*zeta_d)^2)/sqrt((avoidzero(cos(epsilon)-1))^2+(2*zeta_d*cos(epsilon))^2);
      parameter Real phi_ac = atan((2*zeta_s*w_3)/(avoidzero(1-w_3^2)));
      parameter Real phi_y = atan((2*zeta_d*w_1*cos(epsilon))/(avoidzero(cos(epsilon)-w_1^2)))+atan((2*zeta_s*w_3)/avoidzero(1-w_3^2))-atan(2*zeta_d*w_1); 
      parameter Real theta_d = atan((2*zeta_s*w_2)/avoidzero(w_2^2-1));
      parameter Real phi_d = atan((2*zeta_d*cos(epsilon))/avoidzero(cos(epsilon)-1))+atan((2*zeta_s*w_2)/avoidzero(w_2^2-1))-atan(2*zeta_d);
      
      parameter Real X_ac = sqrt(X_acx^2+X_acy^2-2*X_acx*X_acy*cos(phi_ac-phi_y));
      parameter Real X_d = sqrt(X_d1^2+X_d1^2-2*X_d1*X_d2*cos(phi_d-theta_d));
      parameter Real Phi_ac = atan((X_acx*sin(phi_0+phi_ac)-X_acy*sin(phi_0+phi_y))/avoidzero(X_acx*cos(phi_0+phi_ac)-X_acy*cos(phi_0+phi_y)));
      parameter Real Phi_d = atan((X_d1*sin(theta_d)-X_d2*sin(phi_d))/avoidzero(X_d1*cos(theta_d)-X_d2*cos(phi_d)));
      // output
      Real omega_false[3](start={0.0,0.0,0.0}, each fixed=false);
    // internal state for timer, time needs to be incorprated as state due to FMU sim environment
      discrete Integer timer_count(start=0);
    
    algorithm
    // initialize after a loop
      when sample(0, sample_interval) then
        timer_count := timer_count+1;           // [-] increase the count
      end when;
    
    equation
    // gyroscope acoustic noise injection attack (only on yaw rate)
      omega_false[1] = 0.0;
      omega_false[2] = 0.0;
      omega_false[3] = l_g/(4*(dis_d/2)/w_d)*(X_ac*cos((w_ac-w_d)*(sample_interval*pre(timer_count))-Phi_ac)+X_d*cos(Phi_d));
    
    end GyroAcousticAtk;
  end Components;

  package Connectors
    connector IntegerInput = input Integer "'input Integer' as connector" annotation(
      defaultComponentName = "u",
      Icon(graphics = {Polygon(lineColor = {255, 127, 0}, fillColor = {255, 127, 0}, fillPattern = FillPattern.Solid, points = {{-100, 100}, {100, 0}, {-100, -100}, {-100, 100}}), Text(origin = {0, -120}, extent = {{-100, 20}, {100, -20}}, textString = "%name")}, coordinateSystem(extent = {{-100, 100}, {100, -140}}, preserveAspectRatio = true, initialScale = 0.2)),
      Diagram(coordinateSystem(preserveAspectRatio = true, initialScale = 0.2, extent = {{-100, -100}, {100, 100}}), graphics = {Polygon(points = {{0, 50}, {100, 0}, {0, -50}, {0, 50}}, lineColor = {255, 127, 0}, fillColor = {255, 127, 0}, fillPattern = FillPattern.Solid), Text(extent = {{-10, 85}, {-10, 60}}, textColor = {255, 127, 0}, textString = "%name")}),
      Documentation(info = "<html>
  <p>
  Connector with one input signal of type Integer.
  </p>
  </html>"));
    connector IntegerOutput = output Integer "'output Integer' as connector" annotation(
      defaultComponentName = "y",
      Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, 100}, {100, -100}}), graphics = {Polygon(lineColor = {255, 127, 0}, fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid, points = {{-100, 100}, {100, 0}, {-100, -100}, {-100, 100}}), Text(origin = {0, -120}, extent = {{-100, 20}, {100, -20}}, textString = "%name")}),
      Diagram(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100, 100}}), graphics = {Polygon(points = {{-100, 50}, {0, 0}, {-100, -50}, {-100, 50}}, lineColor = {255, 127, 0}, fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid), Text(extent = {{30, 110}, {30, 60}}, textColor = {255, 127, 0}, textString = "%name")}),
      Documentation(info = "<html>
    <p>
    Connector with one output signal of type Integer.
    </p>
    </html>"));
    connector RealOutput = output Real "'output Real' as connector" annotation(
      defaultComponentName = "y",
      Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100, 100}}), graphics = {Polygon(lineColor = {0, 0, 127}, fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid, points = {{-100, 100}, {100, 0}, {-100, -100}, {-100, 100}}), Text(origin = {0, -121}, extent = {{-98, 19}, {98, -19}}, textString = "%name")}),
      Diagram(coordinateSystem(preserveAspectRatio = true, extent = {{-100.0, -100.0}, {100.0, 100.0}}), graphics = {Polygon(lineColor = {0, 0, 127}, fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid, points = {{-100.0, 50.0}, {0.0, 0.0}, {-100.0, -50.0}}), Text(textColor = {0, 0, 127}, extent = {{30.0, 60.0}, {30.0, 110.0}}, textString = "%name")}),
      Documentation(info = "<html>
    <p>
    Connector with one output signal of type Real.
    </p>
    </html>"));
    connector RealInput = input Real "'input Real' as connector" annotation(
      defaultComponentName = "u",
      Icon(graphics = {Polygon(lineColor = {0, 0, 127}, fillColor = {0, 0, 127}, fillPattern = FillPattern.Solid, points = {{-100, 100}, {100, 0}, {-100, -100}, {-100, 100}}), Text(origin = {-1, -120}, extent = {{-101, 20}, {101, -20}}, textString = "%name")}, coordinateSystem(extent = {{-100, -100}, {100, 100}}, preserveAspectRatio = true, initialScale = 0.2)),
      Diagram(coordinateSystem(preserveAspectRatio = true, initialScale = 0.2, extent = {{-100.0, -100.0}, {100.0, 100.0}}), graphics = {Polygon(lineColor = {0, 0, 127}, fillColor = {0, 0, 127}, fillPattern = FillPattern.Solid, points = {{0.0, 50.0}, {100.0, 0.0}, {0.0, -50.0}, {0.0, 50.0}}), Text(textColor = {0, 0, 127}, extent = {{-10.0, 60.0}, {-10.0, 85.0}}, textString = "%name")}),
      Documentation(info = "<html>
    <p>
    Connector with one input signal of type Real.
    </p>
    </html>"));

    expandable connector ControlBus
      parameter Integer nu = 1; // [-] input/command dimension
      Real u[nu];
      annotation(
        Icon(graphics = {Polygon(origin = {0, -20}, points = {{-50, -20}, {-80, 40}, {80, 40}, {50, -20}, {-50, -20}}), Ellipse(origin = {17, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-19, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-51, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {49, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-31, -24}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-1, -24}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {29, -24}, extent = {{-9, 8}, {9, -8}}), Text(origin = {0, -60}, extent = {{-80, 20}, {80, -20}}, textString = "%name")}, coordinateSystem(extent = {{-80, 20}, {80, -80}})),
        Diagram(graphics = {Polygon(origin = {0, -20}, points = {{-50, -20}, {-80, 40}, {80, 40}, {50, -20}, {-50, -20}}), Ellipse(origin = {17, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-19, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-51, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {49, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-31, -24}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-1, -24}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {29, -24}, extent = {{-9, 8}, {9, -8}}), Text(origin = {0, -60}, extent = {{-80, 20}, {80, -20}}, textString = "%name")}, coordinateSystem(extent = {{-80, 20}, {80, -80}})));
    end ControlBus;

    expandable connector SensorBus
      parameter Integer ny = 1; // [-] output/feedback dimension
      Real y[ny];
      annotation(
        Icon(graphics = {Polygon(origin = {0, -20}, points = {{-50, -20}, {-80, 40}, {80, 40}, {50, -20}, {-50, -20}}), Ellipse(origin = {17, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-19, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-51, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {49, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-31, -24}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-1, -24}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {29, -24}, extent = {{-9, 8}, {9, -8}}), Text(origin = {0, -60}, extent = {{-80, 20}, {80, -20}}, textString = "%name")}, coordinateSystem(extent = {{-80, 20}, {80, -80}})),
        Diagram(graphics = {Polygon(origin = {0, -20}, points = {{-50, -20}, {-80, 40}, {80, 40}, {50, -20}, {-50, -20}}), Ellipse(origin = {17, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-19, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-51, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {49, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-31, -24}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-1, -24}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {29, -24}, extent = {{-9, 8}, {9, -8}}), Text(origin = {0, -60}, extent = {{-80, 20}, {80, -20}}, textString = "%name")}, coordinateSystem(extent = {{-80, 20}, {80, -80}})));
    end SensorBus;

    expandable connector RCBus
      parameter Integer nr = 1; // [-] reference/signal dimension
      Real r[nr];
      annotation(
        Icon(graphics = {Polygon(origin = {0, -20}, points = {{-50, -20}, {-80, 40}, {80, 40}, {50, -20}, {-50, -20}}), Ellipse(origin = {17, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-19, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-51, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {49, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-31, -24}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-1, -24}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {29, -24}, extent = {{-9, 8}, {9, -8}}), Text(origin = {0, -60}, extent = {{-80, 20}, {80, -20}}, textString = "%name")}, coordinateSystem(extent = {{-80, 20}, {80, -80}})),
        Diagram(graphics = {Polygon(origin = {0, -20}, points = {{-50, -20}, {-80, 40}, {80, 40}, {50, -20}, {-50, -20}}), Ellipse(origin = {17, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-19, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-51, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {49, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-31, -24}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-1, -24}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {29, -24}, extent = {{-9, 8}, {9, -8}}), Text(origin = {0, -60}, extent = {{-80, 20}, {80, -20}}, textString = "%name")}, coordinateSystem(extent = {{-80, 20}, {80, -80}})));
    end RCBus;

    expandable connector MAVLinkBus
      parameter Integer nr = 1; // [-] reference/signal dimension
      Real r[nr];
      annotation(
        Icon(graphics = {Polygon(origin = {0, -20}, points = {{-50, -20}, {-80, 40}, {80, 40}, {50, -20}, {-50, -20}}), Ellipse(origin = {17, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-19, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-51, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {49, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-31, -24}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-1, -24}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {29, -24}, extent = {{-9, 8}, {9, -8}}), Text(origin = {0, -60}, extent = {{-80, 20}, {80, -20}}, textString = "%name")}, coordinateSystem(extent = {{-80, 20}, {80, -80}})),
        Diagram(graphics = {Polygon(origin = {0, -20}, points = {{-50, -20}, {-80, 40}, {80, 40}, {50, -20}, {-50, -20}}), Ellipse(origin = {17, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-19, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-51, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {49, 0}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-31, -24}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {-1, -24}, extent = {{-9, 8}, {9, -8}}), Ellipse(origin = {29, -24}, extent = {{-9, 8}, {9, -8}}), Text(origin = {0, -60}, extent = {{-80, 20}, {80, -20}}, textString = "%name")}, coordinateSystem(extent = {{-80, 20}, {80, -80}})));
    end MAVLinkBus;
  end Connectors;

  package Constants
    constant Real pi = 3.14159265;
    constant Real g = 9.80665;
    constant Real eps = 1.0e-15;
    constant Real r2d = 180/pi;
    constant Real d2r = pi/180;
    constant Real rpm2radps = 2*pi/60;
    constant Real radps2rpm = 60/2/pi;
  end Constants;

  package Utils
    function quat2rot
      // note: conversion between quaternion and rotational matrix (SO3, dcm)
      //       resulting matrix is not coordinate transformation matrix, rotation matrix (take transpose for coordinate transform)
      // [-] input: quaternion
      input Real q[4];
      // [-] output: rotation matrix
      output Real R[3, 3];
    algorithm
      R[1, 1] := q[1]*q[1] + q[2]*q[2] - q[3]*q[3] - q[4]*q[4];
      R[1, 2] := 2*(q[2]*q[3] - q[1]*q[4]);
      R[1, 3] := 2*(q[2]*q[4] + q[1]*q[3]);
      R[2, 1] := 2*(q[2]*q[3] + q[1]*q[4]);
      R[2, 2] := q[1]*q[1] + q[3]*q[3] - q[2]*q[2] - q[4]*q[4];
      R[2, 3] := 2*(q[3]*q[4] - q[1]*q[2]);
      R[3, 1] := 2*(q[2]*q[4] - q[1]*q[3]);
      R[3, 2] := 2*(q[3]*q[4] + q[1]*q[2]);
      R[3, 3] := q[1]*q[1] + q[4]*q[4] - q[2]*q[2] - q[3]*q[3];
    end quat2rot;

    function clip
      // note: clipping scalar for given upper/lower bounds
      // [-] input: scalar
      input Real x;
      // [-] input: lower bound
      input Real x_min;
      // [-] input: upper bound
      input Real x_max;
      // [-] output: clipped value
      output Real y;
    algorithm
      y := if x < x_min then x_min else if x > x_max then x_max else x;
    end clip;

    function hatmap
      // note: hat map for cross product computation
      // [-] input: three-dimensional vector
      input Real v[3];
      // [-] output: cross prodcut matrix
      output Real V[3, 3];
    algorithm
      V := [0, -v[3], v[2]; v[3], 0, -v[1]; -v[2], v[1], 0];
    end hatmap;

    function quat2eul
      // note: euler angle represents same rotation as quaternion
      // [-] load packages: sine and cosine
      import Modelica.Math.atan;
      import Modelica.Math.asin;
      // [-] input: quaternion
      input Real q[4];
      // [rad] output: euler angle (3-2-1 rotation)
      output Real eul[3];
    algorithm
      eul[1] := atan((2*q[3]*q[4] + 2*q[1]*q[2])/(2*q[1]^2 + 2*q[4]^2 - 1));
      eul[2] := asin(-(2*q[2]*q[4] - 2*q[1]*q[3]));
      eul[3] := atan((2*q[2]*q[3] + 2*q[1]*q[4])/(2*q[1]^2 + 2*q[2]^2 - 1));
    end quat2eul;

    function eul2rot
      // note: euler angle conversion to rotation matrix (roll/pitch/yaw, transpose of coordinate transformation)
      // [-] load packages: sine and cosine
      import Modelica.Math.sin;
      import Modelica.Math.cos;
      // [rad] input: euler angle (3-2-1 rotation)
      input Real euler[3];
      // [-] output: rotation matrix (a point rotation)
      output Real R[3, 3];
    algorithm
      R[1, 1] := cos(euler[2])*cos(euler[3]);
      R[1, 2] := -cos(euler[1])*sin(euler[3]) + sin(euler[1])*sin(euler[2])*cos(euler[3]);
      R[1, 3] := sin(euler[1])*sin(euler[3]) + cos(euler[1])*sin(euler[2])*cos(euler[3]);
      R[2, 1] := cos(euler[2])*sin(euler[3]);
      R[2, 2] := cos(euler[1])*cos(euler[3]) + sin(euler[1])*sin(euler[2])*sin(euler[3]);
      R[2, 3] := -sin(euler[1])*cos(euler[3]) + cos(euler[1])*sin(euler[2])*sin(euler[3]);
      R[3, 1] := -sin(euler[2]);
      R[3, 2] := sin(euler[1])*cos(euler[2]);
      R[3, 3] := cos(euler[1])*cos(euler[2]);
    end eul2rot;

    function quatprod
      // note: quaternion product
      // [-] input: quaternion
      input Real q_in1[4], q_in2[4];
      // [-] output: quaternion
      output Real q_out[4];
    algorithm
      q_out[1] := q_in1[1]*q_in2[1] - q_in1[2]*q_in2[2] - q_in1[3]*q_in2[3] - q_in1[4]*q_in2[4];
      q_out[2] := q_in1[1]*q_in2[2] + q_in1[2]*q_in2[1] + q_in1[3]*q_in2[4] - q_in1[4]*q_in2[3];
      q_out[3] := q_in1[1]*q_in2[3] - q_in1[2]*q_in2[4] + q_in1[3]*q_in2[1] + q_in1[4]*q_in2[2];
      q_out[4] := q_in1[1]*q_in2[4] + q_in1[2]*q_in2[3] - q_in1[3]*q_in2[2] + q_in1[4]*q_in2[1];
    end quatprod;

    function cross3
      input Real v_in1[3], v_in2[3];
      // [-] input vector
      output Real v_out[3];
      // [-] output vector
    algorithm
      v_out[1] := v_in1[2]*v_in2[3] - v_in1[3]*v_in2[2];
      v_out[2] := -(v_in1[1]*v_in2[3] - v_in1[3]*v_in2[1]);
      v_out[3] := v_in1[1]*v_in2[2] - v_in1[2]*v_in2[1];
    end cross3;

    function inv3x3
      // note: inverse of 3x3 matrix using adjugate/determinant
      // [-] input: 3x3 matrix
      input Real M[3,3];
      // [-] output: 3x3 matrix
      output Real invM[3,3];
    protected
      // [-] internal state: determinant and eps for inversion
      Real det;
      constant Real eps = 1e-12;
      
    algorithm
// determinant
      det :=  M[1,1]*(M[2,2]*M[3,3]-M[2,3]*M[3,2])
            - M[1,2]*(M[2,1]*M[3,3]-M[2,3]*M[3,1])
            + M[1,3]*(M[2,1]*M[3,2]-M[2,2]*M[3,1]);
    
      assert(abs(det) > 1e-12,
             "inv3x3: matrix is singular or ill-conditioned (|det| ≈ 0).");
// adjugate (transpose of cofactor) divided by det
      invM[1,1] := (M[2,2]*M[3,3]-M[2,3]*M[3,2])/det;
      invM[1,2] := -(M[1,2]*M[3,3]-M[1,3]*M[3,2])/det;
      invM[1,3] := (M[1,2]*M[2,3]-M[1,3]*M[2,2])/det;
      invM[2,1] := -(M[2,1]*M[3,3]-M[2,3]*M[3,1])/det;
      invM[2,2] := (M[1,1]*M[3,3]-M[1,3]*M[3,1])/det;
      invM[2,3] := -(M[1,1]*M[2,3]-M[1,3]*M[2,1])/det;
      invM[3,1] := (M[2,1]*M[3,2]-M[2,2]*M[3,1])/det;
      invM[3,2] := -(M[1,1]*M[3,2]-M[1,2]*M[3,1])/det;
      invM[3,3] := (M[1,1]*M[2,2]-M[1,2]*M[2,1])/det;
      
    end inv3x3;

    function avoidzero
    
      import GSQuad.Constants;  // [-] load constants
      input Real x;             // [-] input value
      output Real y;                  
    // [-] output value
    algorithm
    
      if abs(x) < Constants.eps then
        y := Constants.eps * sign(x + Constants.eps);   // preserve sign, avoid exact zero
      else
        y := x;
      end if;
      
    end avoidzero;
    annotation(
      Icon);
  end Utils;

  model ExampleHovering
    // parameters
    parameter Integer fidelity = 1 "Select fidelity level" annotation(
      choices(choice = 1 "Low-Fidelity", choice = 2 "High-Fidelity"));
    // load components
    Components.Quadrotor quadrotor(fidelity = fidelity) annotation(
      Placement(transformation(origin = {90, 40}, extent = {{-36, -36}, {36, 36}})));
    Components.Controller controller annotation(
      Placement(transformation(origin = {-32, 38}, extent = {{-37, -37}, {37, 37}})));
    Components.Joystick joystick annotation(
      Placement(transformation(origin = {-162, 60}, extent = {{-35, -35}, {35, 35}})));
    Components.MissionPlanner missionplanner annotation(
      Placement(transformation(origin = {-162, 6}, extent = {{-35, -35}, {35, 35}})));
  equation
    connect(controller.sensor, quadrotor.sensor) annotation(
      Line(points = {{-80, 16}, {-80, -12}, {136, -12}, {136, 40}}, thickness = 0.5));
    connect(quadrotor.control, controller.control) annotation(
      Line(points = {{44, 40}, {16, 40}}, thickness = 0.5));
    connect(controller.rc, joystick.rc) annotation(
      Line(points = {{-80, 54}, {-116, 54}}, thickness = 0.5));
    connect(missionplanner.mavlink, controller.mavlink) annotation(
      Line(points = {{-116, 0}, {-94, 0}, {-94, 33.5}, {-80, 33.5}, {-80, 34}}, thickness = 0.5));
    annotation(
      Diagram(coordinateSystem(extent = {{-200, 100}, {160, -40}})),
      experiment(StartTime = 0.0, StopTime = 30, Tolerance = 1e-06, Interval = 0.005));
  end ExampleHovering;
end GSQuad;
