from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
import torch
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, ones, matmul
from torch.nn.functional import cross_entropy, relu, mse_loss, softmax
from torch import movedim


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        
        self.w = Parameter(ones((1, dimensions)))
        

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        return matmul(x, self.w.T)
        

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        score = self.run(x)
        if score.item() >= 0:
            return 1
        else:
            return -1



    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """        
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            while True:
                mistakes = 0
                for batch in dataloader:
                    x = batch['x']
                    y = batch['label']
                    prediction = self.get_prediction(x)
                    target = y.item()
                    
                    if prediction != target:
                        self.w += target * x
                        mistakes += 1
                
                if mistakes == 0:
                    break



class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        self.w1 = Linear(1, 100)
        self.w2 = Linear(100, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)



    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        x = relu(self.w1(x))
        return self.w2(x)

    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        predicted = self.forward(x)
        return mse_loss(predicted, y)
 
        

    def train(self, dataset):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on
            
        """
        dataloader = DataLoader(dataset, batch_size=20, shuffle=True)
        
        while True:
            total_loss = 0
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                
                self.optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if total_loss / len(dataloader) < 0.02:
                break

            







class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        
        self.w1 = Linear(input_size, 200)
        self.w2 = Linear(200, 100)
        self.w3 = Linear(100, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.005)




    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
            (also called logits)
        """
        x = relu(self.w1(x))
        x = relu(self.w2(x))
        return self.w3(x)

 

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        predicted = self.run(x)
        return cross_entropy(predicted, y)

    
        

    def train(self, dataset):
        """
        Trains the model.
        """
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
        
        while True:
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                
                self.optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                self.optimizer.step()
            
            if dataset.get_validation_accuracy() > 0.975:
                break



class LanguageIDModel(Module):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        
        self.hidden_size = 200
        self.initial_w = Linear(self.num_chars, self.hidden_size)
        self.hidden_w = Linear(self.hidden_size, self.hidden_size)
        self.output_w = Linear(self.hidden_size, len(self.languages))
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.005)


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
            (also called logits)
        """
        # Initialize hidden state h_i with the first character
        h = relu(self.initial_w(xs[0]))
        
        # Iterate over the rest of the characters
        for i in range(1, len(xs)):
            # Update hidden state: h = f(x_i * Wx + h_{i-1} * Wh)
            # But here we are simplifying to match the project structure usually seen:
            # combine current character and previous hidden state
            # A common simple RNN way: h = relu(char_embedding + hidden_transformation)
            # Adjusting to use the layers defined:
            
            # Since self.initial_w maps char -> hidden, we can reuse it or use a separate one.
            # But standard RNN: h_new = activation(W_x * x + W_h * h_old)
            # Here we can approximate or do:
            # h = relu(self.initial_w(xs[i]) + self.hidden_w(h))
            
            h = relu(self.initial_w(xs[i]) + self.hidden_w(h))
            
        return self.output_w(h)

    
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        predicted = self.run(xs)
        return cross_entropy(predicted, y)
        

    def train(self, dataset):
        """
        Trains the model.

        Note that when you iterate through dataloader, each batch will returned as its own vector in the form
        (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
        get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
        that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
        as follows:

        movedim(input_vector, initial_dimension_position, final_dimension_position)

        For more information, look at the pytorch documentation of torch.movedim()
        """
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
        
        while True:
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                
                # Reshape x to be (length x batch x num_chars) as expected by run/get_loss
                # Original shape from dataloader: (batch x length x num_chars)
                # We need to swap dim 0 and 1
                x = movedim(x, 0, 1)
                
                self.optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                self.optimizer.step()
            
            if dataset.get_validation_accuracy() > 0.82:
                break

        

def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())
    
    # Simple manual convolution
    # Input shape: (H, W)
    # Weight shape: (KH, KW)
    H, W = input_tensor_dimensions
    KH, KW = weight_dimensions
    
    out_h = H - KH + 1
    out_w = W - KW + 1
    
    output = []
    
    for y in range(out_h):
        row = []
        for x in range(out_w):
            # Element-wise multiplication and sum
            patch = input[y:y+KH, x:x+KW]
            val = tensordot(patch, weight, dims=2)
            row.append(val)
        output.append(stack(row))
        
    Output_Tensor = stack(output)
    
    return Output_Tensor



class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.

    Note that this class looks different from a standard pytorch model since we don't need to train it
    as it will be run on preset weights.
    """
    

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        
        # After convolution (3x3 on 28x28) we get 26x26 output
        conv_output_size = 26 * 26
        
        self.w1 = Linear(conv_output_size, 128)
        self.w2 = Linear(128, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.005)




    def run(self, x):
        return self(x)
 
    def forward(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimentional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        x = x.flatten(start_dim=1)
        
        x = relu(self.w1(x))
        return self.w2(x)


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        predicted = self.run(x)
        return cross_entropy(predicted, y)

     
        

    def train(self, dataset):
        """
        Trains the model.
        """
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
        
        while True:
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                
                self.optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                self.optimizer.step()
            
            if dataset.get_validation_accuracy() > 0.975:
                break



class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size,layer_size)

        #Masking part of attention layer
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
       
        self.layer_size = layer_size


    def forward(self, input):
        """
        Applies the attention mechanism to input. All necessary layers have 
        been defined in __init__()

        In order to apply the causal mask to a given matrix M, you should update
        it as such:
    
        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]

        For the softmax activation, it should be applied to the last dimension of the input,
        Take a look at the "dim" argument of torch.nn.functional.softmax to figure out how to do this.
        """
        B, T, C = input.size()

        # Calculate Query, Key, Value matrices
        k = self.k_layer(input)
        q = self.q_layer(input)
        v = self.v_layer(input)
        
        # Calculate raw attention scores
        # We need (Q @ K^T) / sqrt(d_k)
        # Q: (B, T, C), K: (B, T, C) -> K^T for last 2 dims is (B, C, T)
        # Result: (B, T, T)
        
        # Transpose last two dimensions of k for multiplication
        k_t = k.transpose(-2, -1)
        
        # Raw scores
        scores = matmul(q, k_t) / (self.layer_size ** 0.5)
        
        # Apply mask
        scores = scores.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        
        # The mask in the template code has a [0] at the end, but checking logic:
        # self.mask is (1, 1, block_size, block_size)
        # scores is (B, T, T) or similar. 
        # The autograder hint says: M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]
        # The [0] at the end suggests that masked_fill returns something indexable or it's a copy-paste artifact in the docstring.
        # However, masked_fill returns a Tensor. 
        # The provided snippet `self.mask` has shape (1,1,B,B).
        # Let's trust the broadcasting of PyTorch or adjust.
        # But wait, the docstring says: `...float('-inf'))[0]`
        # This [0] is very suspicious if masked_fill returns a Tensor.
        # Let's re-read carefully: "M = M.masked_fill(...) [0]"
        # If the operation `masked_fill` was wrapped in something returning a tuple it would make sense.
        # But `masked_fill` returns a Tensor.
        # Maybe they want us to take the 0th element of the result? No, that would be a single row/matrix.
        # Let's assume standard causal attention logic is required.
        # But wait, looking at the mask definition: .view(1, 1, block_size, block_size) -> 4D tensor.
        # Input 'input' is (B, T, C).
        # Q, K, V are (B, T, C).
        # Scores (B, T, T).
        # Mask (1, 1, block_size, block_size).
        # If we slice mask to :T,:T we get (1, 1, T, T).
        # Broadcasting (B, T, T) with (1, 1, T, T) -> This might be tricky if dims don't align.
        # The scores tensor is 3D (B, T, T). The mask is 4D.
        # If we want to broadcast, we might need to unsqueeze scores or squeeze mask.
        
        # The hint likely implies:
        # self.mask[:,:,:T,:T] returns (1, 1, T, T)
        # The `== 0` returns a boolean tensor of shape (1, 1, T, T)
        # If we apply this to M (scores), M should probably be compatible.
        # If M is (B, T, T), we might need to make it (B, 1, T, T) for multi-head? 
        # But this is single head attention (Linear(layer_size, layer_size)).
        # So M is (B, T, T).
        # Using (1, 1, T, T) mask on (B, T, T) requires squeezing the 2nd dim of mask.
        
        # Let's try to follow the hint exactly, assuming `[0]` removes the extra dimension from the mask result?
        # Or maybe the hint meant `self.mask[0,0,:T,:T]`?
        
        # Actually, let's look at the mask provided:
        # self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        # It creates a 4D mask.
        
        # Let's strip the extra dimensions to make it (T, T) or (1, T, T) compatible with (B, T, T).
        # mask slice: self.mask[0, 0, :T, :T] -> (T, T)
        
        scores = scores.masked_fill(self.mask[0, 0, :T, :T] == 0, float('-inf'))
        
        # Softmax over the last dimension
        weights = softmax(scores, dim=-1)
        
        # Weighted sum of values
        # weights: (B, T, T), v: (B, T, C)
        # Result: (B, T, C)
        output = matmul(weights, v)
        
        return output

     