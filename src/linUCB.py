import numpy as np

class MultiArmBandit:

    def __init__(self, N, D):
        """
        Initialize the Multi-Armed Bandit with N arms and D-dimensional context.
        Parameters:
        N (int): Number of arms.
        D (int): Dimension of the context.
        """
        
        self.N = N
        self.D = D

        self.armSet = np.zeros(self.N)
        self.ridgeMatrix = [np.identity(self.D) for _ in range(self.N)]
        self.invRidgeMatrix = [np.identity(self.D) for _ in range(self.N)]
        self.payoffVector = [np.zeros(self.D) for _ in range(self.N)]
        
        self.delta = 0.05 
        self.alpha = 1 + np.sqrt(np.log(2 / self.delta) / 2)


    def selectArm(self, contextVector):
        """
        Select an arm based on the LinUCB algorithm.
        Parameters:
        contextVector (np.ndarray): The context vector of shape (D,).
        Returns:
        int: The index of the selected arm.
        """

        self.invRidgeMatrix = [np.linalg.inv(self.ridgeMatrix[i]) for i in range(self.N)]
        coefficientVector = [self.invRidgeMatrix[i] @ self.payoffVector[i] for i in range(self.N)]

        for i in range(self.N):
            contextVectorT = np.transpose(contextVector)
            estimatedPayoff = contextVectorT @ coefficientVector[i]
            confidenceBound = self.alpha * np.sqrt(contextVectorT @ self.invRidgeMatrix[i] @ contextVector)

            self.armSet[i] = estimatedPayoff + confidenceBound
        
        selectedArm = np.argmax(self.armSet)
        return selectedArm
    

    def updateModel(self, selectedArm, contextVector, payOff):
        """
        Update the model parameters based on the observed reward.
        Parameters:
        context (tuple): A tuple containing the context vector and the selected arm index.
        reward (float): The observed reward for the selected arm.
        """
        
        self.ridgeMatrix[selectedArm] += np.outer(contextVector, contextVector)
        self.payoffVector[selectedArm] += contextVector * payOff
