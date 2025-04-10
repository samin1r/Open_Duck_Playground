import json

# contains only one reference motion


class EpisodicReferenceMotion:
    def __init__(self, ref_motion_path: str):
        self.ref_motion = json.load(open(ref_motion_path, "r"))
        self.nb_steps = len(self.ref_motion["Frames"])

    # ref_motion["Frames"][i]:
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
    def get_frame(self, i):
        # outputs [joints_pos, joints_vel, foot_contacts, base_linear_vel, base_angular_vel],

        frame = []
        frame += self.ref_motion["Frames"][i][7 : 7 + 16]  # joints pos
        frame += self.ref_motion["Frames"][i][35 : 35 + 16]  # joints vel
        frame += self.ref_motion["Frames"][i][-2:]  # foot contacts
        frame += self.ref_motion["Frames"][i][
            29 : 29 + 6
        ]  # base linear vel + base angular vel
        return frame


# if __name__ == "__main__":
#     ERM = EpisodicReferenceMotion("/home/antoine/Téléchargements/animation_data.json")
#     for i in range(ERM.nb_steps):
#         frame = ERM.get_frame(i)
#         exit()
