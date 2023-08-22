#custom variance computation per epoch
class ComputeIterationVaraince:
  def __init__(self,name=""):
    self.name = name
    self.array = []

  def add(self,number):
    self.array.append(number)

  def compute_variance(self):
    numbers = self.array
    n = len(numbers)
    mean = sum(numbers) / n
    variance = sum((x - mean) ** 2 for x in numbers) / n
    return variance

  def reset_states(self):
  	self.array = []