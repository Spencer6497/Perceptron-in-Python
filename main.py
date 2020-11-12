from Perceptron import Perceptron


if __name__ == "__main__":
    train_data = [
        # x1, x2, target
        [1.0, 1.0, 1.0],
        [9.4, 6.4, -1.0],
        [2.5, 2.1, 1.0],
        [8.0, 7.7, -1.0],
        [0.5, 2.2, 1.0],
        [7.9, 8.4, -1.0],
        [7.0, 7.0, -1.0],
        [2.8, 0.8, 1.0],
        [1.2, 3.0, 1.0],
        [7.8, 6.1, -1.0]
    ]

    test_data = [
        # x1, x2
        [8.7, 1.0],
        [2.5, 2.3],
        [2.5, 2.0],
        [1.1, 2.4],
        [8.0, 2.2],
        [9.5, 10.0],
        [1.3, 7.0],
        [2.8, 6.2],
        [8.0, 8.8],
        [6.2, 6.1]
    ]

    # Instantiate perceptron
    percep = Perceptron()
    # Train the perceptron on the training data
    percep.train(train_data)
    # Predict values of test data based on the training model
    percep.predict(test_data)
