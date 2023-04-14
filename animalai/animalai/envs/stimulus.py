
class Stimulus():
    def __init__(self, agent, list_of_objects, raycasts, reward=None):
        self.agent = agent
        self.list_of_objects = list_of_objects
        self.reward = reward

        # build stimulus from parsed raycast
        self.objs_left = []
        self.objs_ahead = []
        self.objs_right = []

        parsed_raycasts = self.agent.raycast_parser.parse(raycasts)

        for obj in self.list_of_objects:
            if self.agent.left(parsed_raycasts, obj):
                self.objs_left.append(obj)

            if self.agent.ahead(parsed_raycasts, obj):
                self.objs_ahead.append(obj)

            if self.agent.right(parsed_raycasts, obj):
                self.objs_right.append(obj)

        self.objs_left = sorted(self.objs_left, key=lambda x: x.name)
        self.objs_ahead = sorted(self.objs_ahead, key=lambda x: x.name)
        self.objs_right = sorted(self.objs_right, key=lambda x: x.name)

        # unconditioned value is assumed to be constant
        if self.reward is not None:
            self.u_val = self.reward
        else:
            self.u_val = 0

    def __hash__(self):
        aux = list(self.objs_left)
        aux.extend(list(self.objs_ahead))
        aux.extend(list(self.objs_right))
        aux.append(self.u_val)
        return hash(tuple(aux))

    def __eq__(self, o):
        if self.u_val != o.u_val:
            # should only be equal if they're both 0
            return False

        if self.objs_left != o.objs_left:
            return False
        if self.objs_ahead != o.objs_ahead:
            return False
        if self.objs_right != o.objs_right:
            return False

        return True

    def __str__(self):
        return ("---------------------------------------------\n" +
                "Unconditioned value: %.4f\n" % self.u_val +
                "Left: " + ",".join(map(lambda x: x.name, self.objs_left)) + "\n" +
                "Ahead: " + ",".join(map(lambda x: x.name, self.objs_ahead)) + "\n" +
                "Right: " + ",".join(map(lambda x: x.name, self.objs_right)) + "\n" +
                "---------------------------------------------\n")

    def update_stimulus_value(self, next_s):
        self.val += self.agent.alpha_w * (next_s.u_val + next_s.val - self.val)
