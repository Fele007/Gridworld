

class Gridworld(object):
    def __init__(self, height, width, winning_field, losing_field, illegal_fields, start_field):
        """ Creates a gridworld with specified dimensions and win/lose properties,
        where fields are indexed from 1 upper left to width * height lower right 
        Params:
        illegal_fields: expects set of fields """
        
        #TODO: Sanity check for fields

        self.height = height
        self.width = width
        self.state_space = [k for k in range(self.height * self.width + 1)]
        self.action_space = [1,2,3,4] # in order UP LEFT DOWN RIGHT like WASD
        self.winning_field = winning_field 
        self.losing_field = losing_field   
        self.legal_fields = set(self.state_space) - illegal_fields
        self.start_field = start_field
        self.field = start_field

    def reachable_states(self, field):
        """ States that are reachable (in theory) given the action space but are not neccessarily legal """

        return [field - 1, field + 1, field + self.width, field - self.width]

    def calculate_reward(self, field_violation):
        if self.field == self.winning_field:
            return 10
        elif self.field == self.losing_field:
            return -10
        elif field_violation:
            return -1
        else:
            return 0

    def calculate_field(self, action):
        field_violation = False
        field = self.field
        if action == 1:
            field = self.field - self.width
        elif action == 2:
            field = self.field - 1
            if field % self.width == 0:
                field_violation = True
        elif action == 3:
            field = self.field + self.width
        elif action == 4:
            field = self.field + 1
            if self.field % self.width == 0:
                field_violation = True
        if field in self.legal_fields and not field_violation:
            return field, field_violation
        else:
            return self.field, field_violation

    def calculate_done(self):
        if self.field == self.winning_field or self.field == self.losing_field:
            self.reset()
            return True
        else:
            return False

    def step(self, action):
        """ Returns observation (field), reward and done/not done yet for an agent to deal with 
        Params:
        action: 1 -> Up, 2 -> Left, 3 -> Down, 4 -> Right """

        self.field, field_violation = self.calculate_field(action)
        return self.field, self.calculate_reward(field_violation), self.calculate_done()

    def reset(self):
        """ Initializes the Gridworld to start state """

        self.field = self.start_field
