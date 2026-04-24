class Interpreter:

    def interpret(self, features, label):

        if label == "normal":
            return "normal"

        # check push-off weakness
        if features["second_peak"] < 0.9:
            return "weak_push_off"

        return "other_impairment"