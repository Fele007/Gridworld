class Gridworld(object):
    def __init__(self, height, width, winning_field, losing_field, illegal_fields, start_field):
        """ Creates a gridworld with specified dimensions and win/lose properties,
        where fields are indexed from 1 upper left to width * height lower right 
        Params:
        illegal_fields: expects set of fields """

        self.height = height
        self.width = width
        self.state_space = [[k] for k in range(1, self.height * self.width + 1)]
        self.action_space = [0,1,2,3] # in order UP LEFT DOWN RIGHT like WASD
        self.winning_observation = [winning_field] 
        self.losing_observation = [losing_field]   
        self.legal_observations = set(self.state_space) - illegal_fields
        self.start_observation = [start_field]
        self.observation = start_field

    def reachable_states(self, field):
        """ States that are reachable (in theory) given the action space but are not neccessarily legal """

        return [field - 1, field + 1, field + self.width, field - self.width]

    def calculate_reward(self, field_violation):
        if self.observation == self.winning_observation:
            return 10
        elif self.observation == self.losing_observation:
            return -10
        elif field_violation:
            return -1
        else:
            return 0

    def calculate_observation(self, observation, action):
        field_violation = False
        field = observation
        if action == 0:
            field = observation - self.width
        elif action == 1:
            field = observation - 1
            if field % self.width == 0:
                field_violation = True
        elif action == 2:
            field = observation + self.width
        elif action == 3:
            field = observation + 1
            if observation % self.width == 0:
                field_violation = True
        if field in self.legal_observations and not field_violation:
            return field, field_violation
        else:
            return observation, field_violation

    def calculate_done(self):
        if self.observation == self.winning_observation or self.observation == self.losing_observation:
            self.reset()
            return True
        else:
            return False

    def step(self, action):
        """ Returns observation (field), reward and done/not done yet for an agent to deal with 
        Params:
        action: 0 -> Up, 1 -> Left, 2 -> Down, 3 -> Right """

        self.observation, field_violation = self.calculate_observation(self.observation, action)
        return self.observation, self.calculate_reward(field_violation), self.calculate_done()

    def reset(self):
        """ Initializes the Gridworld to start state """

        self.observation = self.start_observation

    def show(self):
        print()
        for field in self.state_space:
            if field % self.width == 1:
                print("", end="|")
            if field == self.observation:
                print("AA", end="|")
            elif field not in self.legal_observations:
                print("XX", end="|")
            elif field == self.winning_observation:
                print("WW", end="|")
            elif field == self.losing_observation:
                print("LL", end="|")
            elif int(field / 10) == 0:
                print(" " + str(field), end="|")
            else:
                print(field, end="|")
            if field % self.width == 0:
                print("")
