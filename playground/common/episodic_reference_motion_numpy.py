import numpy as np
import json

# contains only one reference motion


class EpisodicReferenceMotion:
    def __init__(self, ref_motion_path: str):
        self.ref_motion = json.load(open(ref_motion_path, "r"))
        self.nb_steps = len(self.ref_motion["Frames"])


    # root_position
    # + root_orientation_quat
    # + joints_positions
    # + left_toe_pos
    # + right_toe_pos
    # + world_linear_vel
    # + world_angular_vel
    # + joints_vel
    # + left_toe_vel
    # + right_toe_vel
    # + foot_contacts
    # [joints_pos, joints_vel, foot_contacts, base_linear_vel, base_angular_vel],
    def get_frame(self, i):
        # frame_dict = self.ref_motion["Frames"][i]
        # print(self.ref_motion["Frames"][i])
        joints_pos = self.ref_motion["Frames"][i][7:7+14]
        # joints_pos = np.array(frame_dict["joints_positions"])
        joints_pos = list(np.insert(joints_pos, 9, [0, 0]))

        
        frame = []
        frame += joints_pos
        frame += list(np.zeros(24))
        # frame += list(frame_dict["joints_vel"]) + [0, 0]
        # frame += [0, 0]
        # frame += list(frame_dict["world_linear_vel"])
        # frame += [0, 0, 0]
        
        # for val in frame_dict.values():
        #     frame += list(val)
        # frame += [0, 0, 0, 0, 0]
        # # print(len(frame))
        return frame


if __name__ == "__main__":
    ERM = EpisodicReferenceMotion("/home/antoine/Téléchargements/animation_data.json")
    for i in range(ERM.nb_steps):
        frame = ERM.get_frame(i)
        exit()
        # print(f"Frame {i}:")
        # print(frame)
        # print("==")
        # # print(f"  qpos: {frame['qpos']}")
        # # print(f"  ctrl: {frame['ctrl']}")
        # # print(f"  time: {frame['time']}")
