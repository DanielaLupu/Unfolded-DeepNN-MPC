
import torch

# Define Layers
class PfirstLayer(torch.nn.Module):
  def __init__(self, size_in, size_out, size_state, u_lb, u_ub):
    super().__init__()
    self.size_in, self.size_out, self.size_state = size_in, size_out, size_state
    self.lb, self.ub = u_lb, u_ub

    self.Q1 = torch.nn.Parameter(torch.zeros(self.size_in, self.size_in))
    self.Q2 = torch.nn.Parameter(torch.zeros(self.size_in, self.size_state))

  def forward(self, initState):
    #cmd = self.lb + (self.ub-self.lb) * torch.rand(self.size_in,1, dtype= torch.double)   # initialize a random u0=u1
    cmd = torch.zeros(self.size_in, 1, dtype= torch.double)
    matrix1 = torch.add(torch.eye(self.size_in), self.Q1, alpha = -1)
    term1 = torch.mm(matrix1, cmd)

    b =torch.mm(self.Q2, initState.T)
    cmdNext = torch.add(term1, b, alpha = -1)

    fctAct = torch.nn.Hardtanh(self.lb, self.ub)

    return (cmd, fctAct(cmdNext), initState)

# ----------------------------------------------------------------------------------

class PunfoldingLayer(torch.nn.Module):

    def __init__(self, size_in, size_out, size_state, u_lb, u_ub):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out     # size_in = dim of comand
        self.lb, self.ub = u_lb, u_ub 
        self.size_state = size_state

        self.Q1 = torch.nn.Parameter(torch.zeros(self.size_in, self.size_in))
        self.Q2 = torch.nn.Parameter(torch.zeros(self.size_in, self.size_state))

        self.beta = torch.nn.Parameter(torch.zeros((1)))


    def forward(self, input):

        cmdPrev = input[0]
        cmdCurrent = input[1]
        state = input[2]

        matrix1 = self.beta * torch.add( torch.eye(self.size_in), self.Q1, alpha =-1)
        term1 = torch.mm(matrix1, cmdPrev)

        matrix2 = (1+self.beta) * torch.add(torch.eye(self.size_in), self.Q1, alpha = -1)
        term2 = torch.mm(matrix2, cmdCurrent)

        bias = torch.mm(self.Q2, state.T)
        term12 = torch.add(term2, term1, alpha = -1 )
        nextCmd = torch.add(term12, bias, alpha = -1)

        fctAct = torch.nn.Hardtanh(self.lb,self.ub)
        return (cmdCurrent, fctAct(nextCmd), state)

# ---------------------------------------------------------------------------------
class outLayer(torch.nn.Module):

  def __init__(self, size_in, size_out):
    super().__init__()
  def forward(self, input):
    return input[1]

# ---------------------------------------------------------------------------------

# Define Layers
class PDfirstLayer(torch.nn.Module):
  def __init__(self, size_in, size_out, size_state, size_horz, u_lb, u_ub, z_lb, z_ub): #size_in = T*dim(u), size_state = dim(z), size_horz = T
    super().__init__()
    self.size_in, self.size_out, self.size_state = size_in, size_out, size_state
    self.size_horz = size_horz
    self.lb, self.ub =  u_lb, u_ub

    self.H = torch.nn.Parameter(torch.zeros(self.size_in, self.size_in))          # H :Nnu x Nnu
    self.Q1 = torch.nn.Parameter(torch.zeros( self.size_horz*self.size_state, self.size_in))     # A (or \bar(AB)): Nnz x Nnu
    self.Q2 = torch.nn.Parameter(torch.zeros( self.size_horz*self.size_state, self.size_state))                 # AN: Nnz xnz
    self.Q3 = torch.nn.Parameter(torch.zeros(self.size_in, self.size_horz*self.size_state))     # A.TQ from q(x0) : Nnu x Nnz
    self.Q4 = torch.nn.Parameter(torch.ones( self.size_in, self.size_horz*self.size_state))

    self.lbd = torch.nn.Parameter(torch.ones((1)))   # lambda
    self.gamma = torch.nn.Parameter(torch.ones((1)))   # gamma
    #self.tau = torch.nn.Parameter(torch.empty((1)))   # tau  = lambda/gamma
    #Constants from the satate constraints
    self.Cx = torch.kron(torch.eye(self.size_horz), torch.cat((torch.eye(self.size_state), -torch.eye(self.size_state)), dim=0)).to(torch.double)

    # Concatenate z_ub with itself along dimension 0 to create a tensor of shape (12,)
    z_ub_concat = torch.cat((z_ub.to(torch.double), -z_lb.to(torch.double)), dim=0)
    # Use the Kronecker product with a column vector of ones to repeat the pattern 5 times
    self.dx = torch.kron(torch.ones(self.size_horz, 1), z_ub_concat.view(-1, 1))

    #Initialize ReLU
    self.relu = torch.nn.ReLU()
    self.hardT = torch.nn.Hardtanh(self.lb,self.ub)
  def forward(self, initState):
    cmd = torch.ones(self.size_in, 1, dtype= torch.double)   # initialize a random u0=u1
    miu = torch.ones(2*self.size_horz*self.size_state, 1, dtype= torch.double)   # dual variable miu_k

   # miu update
    dvar1 = self.Cx@self.Q1@cmd
    dvar2 = self.Cx@self.Q2 @initState.T
    pond = self.lbd/self.gamma
    dvar = miu +  pond* (dvar1- self.dx - dvar2);
    miu_new = self.relu(dvar)

    # state update
    pvar1 = self.Q4 @ self.Cx.T@ miu_new
    cmdNext = cmd -self.gamma* (self.H @cmd + self.Q3@ self.Q2 @ initState.T + pvar1)

    return (cmd, self.hardT(cmdNext), miu_new, initState)

# ----------------------------------------------------------------------------------

class PDunfoldingLayer(torch.nn.Module):

     def __init__(self, size_in, size_out, size_state, size_horz, u_lb, u_ub, z_lb, z_ub):  #size_in = T*dim(u), size_state = dim(z), size_horz = T
       super().__init__()
       self.size_in, self.size_out, self.size_state = size_in, size_out, size_state
       self.size_horz = size_horz
       self.lb, self.ub =  lb, ub

       self.H = torch.nn.Parameter(torch.zeros(self.size_in, self.size_in))                         # H :Nnu x Nnu
       self.Q1 = torch.nn.Parameter(torch.zeros( self.size_horz*self.size_state, self.size_in))     # A (or \bar(AB)): Nnz x Nnu
       self.Q2 = torch.nn.Parameter(torch.zeros( self.size_horz*self.size_state, self.size_state))  # AN: Nnz xnz
       self.Q3 = torch.nn.Parameter(torch.zeros( self.size_in, self.size_horz*self.size_state))     # A.TQ from q(x0) : Nnu x Nnz
       self.Q4 = torch.nn.Parameter(torch.ones( self.size_in, self.size_horz*self.size_state))

       self.lbd = torch.nn.Parameter(torch.ones((1)))   # lambda
       self.gamma = torch.nn.Parameter(torch.empty((1)))   # gamma
       #self.tau = torch.nn.Parameter(torch.empty((1)))   # tau = lambda/gamma

       #Constants from the satate constraints
       self.Cx = torch.kron(torch.eye(self.size_horz), torch.cat((torch.eye(self.size_state), -torch.eye(self.size_state)), dim=0)).to(torch.double)
       # Concatenate z_ub with itself along dimension 0 to create a tensor of shape (12,)
       z_ub_concat = torch.cat((z_ub.to(torch.double), -z_lb.to(torch.double)), dim=0)
       # Use the Kronecker product with a column vector of ones to repeat the pattern 5 times
       self.dx = torch.kron(torch.ones(self.size_horz, 1), z_ub_concat.view(-1, 1))
       
       #Initialize ReLU
       self.relu = torch.nn.ReLU()
       self.hardT = torch.nn.Hardtanh(self.lb,self.ub)


     def forward(self, input):

        cmdPrev = input[0]
        cmdCurrent = input[1]
        miuCurrent = input[2]
        state0 = input[3]

         # miu update
        dvar1 = self.Cx@self.Q1@ (2*cmdCurrent -cmdPrev)
        dvar3 = self.Cx@self.Q1@ self.H @ (2*cmdCurrent -cmdPrev)
        dvar2 = self.Cx@self.Q2 @state0.T
        pond = self.lbd/self.gamma
        dvar = miuCurrent +  pond* (dvar1 -self.dx - dvar2) - self.lbd *dvar3;
        miu_new = self.relu(dvar)

        # state update
        pvar1 = self.Q4 @ self.Cx.T@ miu_new
        cmdNext = cmdCurrent -self.gamma* (self.H @cmdCurrent + self.Q3@ self.Q2 @ state0.T + pvar1)

        return (cmdCurrent, self.hardT(cmdNext), miu_new,state0)

# -----------------------------------------------------------------------------
