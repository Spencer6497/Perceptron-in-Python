class Perceptron:
    # Class constructor
    def __init__(self):
        self.bias = 1
        # Randomly initialize weights (x1 weight, x2 weight, bias weight)
        self.weights = [0.5, -0.6, 0.2]
        # Define learning constant
        self.c = 0.2
        # Define # samples, used in error calculations
        self.numSamples = 0
        # Define # errors, used in error calculations
        self.numErrors = 0

    # Define activation function (linear)
    def activation(self, x):
        return 1 if x >= 0 else -1

    # Define adjust weight method
    def adjust(self, ithLine, diff):
        self.weights[0] += (self.c * diff * ithLine[0])
        self.weights[1] += (self.c * diff * ithLine[1])
        self.weights[2] += (self.c * diff)

    # Define predictive method
    def predict(self, data):
        print("\nPredictions:")
        for sample in data:
            # Calculate summation
            summation = (sample[0] * self.weights[0]) + (sample[1] * self.weights[1]) + self.weights[2]
            # Print prediction
            print(self.activation(summation))

    # Define training method
    def train(self, data):
        for sample in data:
            # Increment # samples
            self.numSamples += 1
            # Calculate summation
            summation = (sample[0] * self.weights[0]) + (sample[1] * self.weights[1]) + self.weights[2]
            # Pass summation to activation function
            act = self.activation(summation)
            # If actual and desired output differ, adjust weights accordingly
            if act != sample[2]:
                # Increment # errors
                self.numErrors += 1
                self.adjust(sample, sample[2] - act)

        print("\nThe accuracy of this perceptron is %.2f%%." % ((self.numSamples - self.numErrors) / self.numSamples * 100))