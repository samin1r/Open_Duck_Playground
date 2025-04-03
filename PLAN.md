- [X] Validate that main is still the best transferable policy
- [X] Can action_max_delay be reduced to 2 ? NO
- [X] Can action_max_delay be reduced to 1 ? NO
- [] Does using mansin's weights in the imitation reward :
  - Improve the simulation behavior (walking nicely in all directions) -> kind of, but it moves more
  - Still transfers -> transfer is less good, but not catastrophic? 
```
    w_torso_pos = 0.0
    w_torso_orientation = 0.0
    w_lin_vel_xy = 0.0
    w_lin_vel_z = 0.0
    w_ang_vel_xy = 0.0
    w_ang_vel_z = 0.0
    w_joint_pos = 15.0
    w_joint_vel = 1.0e-3
    w_contact = 5.0
```
- [] Do we use too much randomization ? 
- 
# Taming the head
Preventing the policy from using the head seems to degrade a lot the policy. 
First, we want a nice walk with the head moving less
Second, we want to be able to control the head position. Two ways of doing that
  - The policy doesn't control the head dofs, we train with randomized positions and it learns to adapt
  - The head dofs position is part of the command and we have a reward to track the desired head dofs. The policy controls the head dofs
- [] Try a very small penalty for moving the head too much (based on velocity)