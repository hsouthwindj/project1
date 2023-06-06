import torch
import numpy as np
import random
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from scipy.special import softmax

from scipy import spatial

# generate similarity vector from given vector and cosine similarity
def torch_cos_sim(v,cos_theta,n_vectors = 1,EXACT = True):
    """
    EXACT - if True, all vectors will have exactly cos_theta similarity. 
           if False, all vectors will have >= cos_theta similarity
    v - original vector (1D tensor)
    cos_theta -cos similarity in range [-1,1]
    """
    # unit vector in direction of v
    u = v / torch.norm(v)
    u = u.unsqueeze(0).repeat(n_vectors,1)

    # random vector with elements in range [-1,1]
    r = torch.rand([n_vectors,len(v)])*2 -1 
    r = r.cuda()

    # unit vector perpendicular to v and u
    uperp = torch.stack([r[i] - (torch.dot(r[i],u[i]) * u[i]) for i in range(len(u))])
    uperp = uperp/ (torch.norm(uperp,dim = 1).unsqueeze(1).repeat(1,v.shape[0]))

    if not EXACT:
        cos_theta = torch.rand(n_vectors)* (1-cos_theta) + cos_theta
        cos_theta = cos_theta.unsqueeze(1).repeat(1,v.shape[0])       
        cos_theta = cos_theta.cuda()
    # w is the linear combination of u and uperp with coefficients costheta
    # and sin(theta) = sqrt(1 - costheta**2), respectively:
    w = cos_theta*u + torch.sqrt(1 - torch.tensor(cos_theta).cuda()**2)*uperp

    return w
# shape [N, fn, 3, 64, 64]
def get_rand_dir(v):
    arr = v.shape
    for i in range(arr[0]):
        temp = v[i,0].flatten()
        for j in range(arr[1]):
            v[i, j,:,:,:] = torch_cos_sim(temp, 0.8,EXACT = False).reshape(1, 3, 64, 64)
    return v
            

def group_cs_batch(bvs):
    re = torch.zeros(len(bvs))
    for i in range(len(re)):
        total = sum(bvs[i])
        re[i] = sum([-1 * (spatial.distance.cosine(total.flatten().detach().cpu(), bvs[i][j].flatten().detach().cpu()) - 1) for j in range(len(bvs[i]))])
    # print(re.shape)
    return re.reshape(len(re),1).cuda()
    # shape [N, 10, 3, 64, 64]
    # return sum([-1 * (spatial.distance.cosine(total.flatten().detach().cpu(), vs[i].flatten().detach().cpu()) - 1) for i in range(len(vs))])
    

def pred(model, batch, model_type):
    preds = []
    if model_type == 'rnn':
        pred, _ = model(batch)
        pred = torch.sigmoid(pred)
        # for i in range(len(batch)):
        #     print(i)
        #     pred, _ = model(batch[i].unsqueeze(0))
        #     pred = torch.sigmoid(pred)
        #     preds.append(pred)
    else:
        for i in range(len(batch)):
            # data = batch[i].squeeze(0) # maybe need this ??
            pred = model(data)
            pred = torch.sigmoid(pred)
            pred = pred.mean() # make it return one number only
            preds.append(preds)
    return pred


def is_attack(model, data, t, model_type = 'rnn'):
    """
    Inputs
    model: the PyTorch NN model
    data: pertubation + original sample
    t: true class label
    Output
    adv_attack: Whether the new sample is an adversarial attack
    """
    adv_attack = False
    model.eval()
    if model_type == 'rnn':
        t_out, _ = model(data)
        t_out = torch.sigmoid(t_out)
    else:
        pass
    if t_out > 0.5:
        adv_attack = False
    else:
        adv_attack = True

    return adv_attack

def gen_attack(N, x_orig, t, delta_max, alpha_min, rho_min, G, num_elite, model, fitness_fun, device, model_type):
    # Algo 1 from : https://arxiv.org/pdf/1805.11090.pdf
    #Convergence - best fitness difference within episilon for n generations 
    #
    """
        N : size of population
        x_orig: original example (batch, channel, x, y), batch = 1
        t: true label
        delta_max: maxmimum distance 
        alpha_min: min mutation range (~15%)
        rho_min: min mutation probability (~10%)
        G: # of generations
        num_elites: number of top members to keep for the next generation
        model: the attacked model
        fitness_fun: the objective function used to calculate the fitness
        device: hardware the tensors will be stored on
    """
    # initialize population
    # vid input shape [1, frame_num, 3, x, y]
    dims = list(x_orig.size())
    dims[0] = N
    # population is an (N, 1, 28, 28) in the case of MNIST
    population = torch.empty(dims, device=device).uniform_(-delta_max, delta_max)
    population = torch.clamp(population + x_orig, 0, 1) - x_orig

    #initialize varaibles used in while loop
    count = 1          #Start with an intial population - so count starts as 1
    crit = 1e-5
    adv_attack = last_best = num_i = num_plat = 0

    #Continue until max num. of iterations or get an adversarial example
    while adv_attack != True and count < G:
        print(count)
        if count % 100 == 0:
            print("Generation " + str(count))
      # Find fitness for every individual and save the best fitness
        fitness = fitness_fun(model, population + x_orig, t, model_type, population)
        best_fit = min(fitness)

      #Check to if fitness last two generations is the same, update num_plat
        if abs(best_fit - last_best) <= crit:
            num_i += 1
            if num_i % 100 == 0 and num_i != 0:
                print("Plateau at Generation " + str(count))
                num_plat += 1
        else:
            num_i = 0

      # TODO: This block sorts the population by fitness,
      # can we use this:
      # new_pop = population.clone()[sorted_inds]
      # or
      # new_pop = population.clone()[fitness.argsort()]
      
      # TODO: we can use sorted_inds = fitness.argsort() instead for simplicity
      # Get sorted indices (Ascending!)
        _, sorted_inds = fitness.sort() 
      #Initialize new population by adding the elites
        new_pop = torch.zeros_like(population)
        for i in range(num_elite):
            new_pop[i] = population[sorted_inds[i]]
      
        #The best individual is the one with the best fitness
        best = new_pop[0]
        print('best', pred(model, best.unsqueeze(0), model_type))
        print('worst', pred(model, population[sorted_inds[N - 1]].unsqueeze(0), model_type))
      
        adv_attack = is_attack(model, best + x_orig, t)
      #If not a true adversarial example need to go to next generation
        if adv_attack == False:
            alpha = max(alpha_min, 0.5 * (0.9 ** num_plat))
            rho = max(rho_min, 0.4 * (0.9 ** num_plat))
            # Softmax fitnesses
            soft_fit = F.softmax(-fitness, dim=0) # Negate fitness since we're trying to minimize
            #need to get apply selection and get a new population
            child_pop = selection(population, soft_fit, x_orig, count, alpha, rho, delta_max, num_elite)
            del population
            new_pop[num_elite:] = child_pop
            population = new_pop
            count += 1
            ## Need to retain best fitness
            last_best = best_fit

    return best, adv_attack, count

def mutation_op(cur_pop, x_orig, idx, alpha=0.15, rho=0.1, delta_max=0.1):
    """
        cur_pop: the current population
        x_orig :  the image we are using for the attack
        idx    : an index. useful for debugging
        alpha: mutation range
        rho: mutation probability
        delta_max: maxmimum distance
    """
    step_noise = alpha * delta_max
    
    perturb_noise = torch.empty_like(cur_pop).uniform_(-step_noise, step_noise)
    mask = torch.empty_like(cur_pop).bernoulli_(rho)
    mutated_pop = perturb_noise * mask + cur_pop
    clamped_mutation_pop = torch.clamp(mutated_pop + x_orig, 0, 1) - x_orig
    normalized_pop = torch.clamp(clamped_mutation_pop, -delta_max, delta_max)
    return normalized_pop

# x = torch.FloatTensor(2, 2)
# image = torch.FloatTensor(2, 2).uniform_(0, 1)
# print(x, image)
# mutpop = mutation_op(x, image, 1)

def fitness(model, batch, target_class, model_type, perts):
    """
        model: a pytorch neural network model (takes x to log probability)
        batch: a batch of examples
        target_class: the class label of x (as an integer)
        returns: tensor of fitnesses
        This source code is inspired by:
        https://github.com/nesl/adversarial_genattack/blob/2304cdc2a49d2c16b9f43821ad8a29d664f334d1/genattack_tf2.py#L39
    """
    # TODO: does it seem wonky to anyone that we are passing in pop + x_orig
    # into the fitness? this might be a design flaw of ours
    model.eval()
    with torch.no_grad():
        probs = pred(model, batch, model_type)
        s = torch.expm1(probs)
        f = probs + s
        print(f.shape)
        f = f + 0.1 * group_cs_batch(perts)
        # s = - torch.expm1(log_probs[:, target_class]) # Sum of probabilities, minus target class, might be more numerically stabe
        # f = log_probs[:, target_class] - torch.log(s)
        return torch.clamp(f.flatten(), -1000, 1000) # clamping to avoid the "all inf" problem

def selection(population, soft_fit, data, idx, alpha, rho, delta_max, num_elite = 2):
    """
        Input
        population: the population of individuals
        soft_fit: the softmax of the fitnesses 
        data: the input value to find a perturbation for
        idx: the generation number we're on - for debugging purposes
        alpha: mutation range
        rho: mutation probability
        num_elite: the number of elites to carry on from the previous generations
        Output
        mut_child_pop: Returns the mutated population of children
    """

    # Crossover
    cdims = list(population.size())
    child_pop_size = population.size()[0] - num_elite
    cdims[0] = child_pop_size
    child_pop = torch.empty(cdims, device=data.device)
    # Roulette
    roulette = Categorical(probs=soft_fit)
    for i in range(child_pop_size):
        parent1_idx = roulette.sample()
        soft_fit_nop1 = soft_fit.clone() + 0.0001 # Incrementing by epsilon to avoid the "all zeros" problem
        soft_fit_nop1[parent1_idx] = 0
        roulette2 = Categorical(probs=soft_fit_nop1)
        parent2_idx = roulette2.sample()
        child = crossover(population[parent1_idx], population[parent2_idx], soft_fit[parent1_idx], soft_fit[parent2_idx])
        child_pop[i] = child
    
    # Mutation
    mut_child_pop = mutation_op(child_pop, data, idx, alpha, rho, delta_max)

    return mut_child_pop

def crossover(parent1, parent2, p1, p2):
    """
    Element-wise crossover
    parent1: individual in old population
    parent2: individual in old population
    p1: softmaxed fitness for parent1
    p2: softmaxed fitness for parent2
    output
    child: new individual from mating of parents
    """
    p = p1/(p1 + p2)
    mask = torch.empty_like(parent1).bernoulli_(p)
    child = mask * parent1 + (1 - mask) * parent2
    return child

def cgen_attack(N, x_orig, t, delta_max, alpha_min, rho_min, G, num_elite, model, fitness_fun, device, model_type):
    """
        N : size of population
        x_orig: original example (batch, channel, x, y), batch = 1
        t: true label
        delta_max: maxmimum distance 
        alpha_min: min mutation range (~15%)
        rho_min: min mutation probability (~10%)
        G: # of generations
        num_elites: number of top members to keep for the next generation
        model: the attacked model
        fitness_fun: the objective function used to calculate the fitness
        device: hardware the tensors will be stored on
    """
    # initialize population
    # vid input shape [1, frame_num, 3, x, y]
    sh = list(x_orig.size())[2:]
    frame_num = list(x_orig.size())[1]
    sh = [N] + sh # population number
    direction = torch.empty(sh, device = device).uniform_(-delta_max, delta_max)
    population = direction.repeat(frame_num, 1,1,1,1).permute(1, 0, 2, 3, 4)
    population = get_rand_dir(population)
    
    
    # population is an (N, 1, 28, 28) in the case of MNIST
    population = torch.empty(dims, device=device).uniform_(-delta_max, delta_max)
    population = torch.clamp(population + x_orig, 0, 1) - x_orig

    #initialize varaibles used in while loop
    count = 1          #Start with an intial population - so count starts as 1
    crit = 1e-5
    adv_attack = last_best = num_i = num_plat = 0

    #Continue until max num. of iterations or get an adversarial example
    while adv_attack != True and count < G:
        print(count)
        if count % 100 == 0:
            print("Generation " + str(count))
      # Find fitness for every individual and save the best fitness
        fitness = fitness_fun(model, population + x_orig, t, model_type, population)
        best_fit = min(fitness)

      #Check to if fitness last two generations is the same, update num_plat
        if abs(best_fit - last_best) <= crit:
            num_i += 1
            if num_i % 100 == 0 and num_i != 0:
                print("Plateau at Generation " + str(count))
                num_plat += 1
        else:
            num_i = 0

      # TODO: This block sorts the population by fitness,
      # can we use this:
      # new_pop = population.clone()[sorted_inds]
      # or
      # new_pop = population.clone()[fitness.argsort()]
      
      # TODO: we can use sorted_inds = fitness.argsort() instead for simplicity
      # Get sorted indices (Ascending!)
        _, sorted_inds = fitness.sort() 
      #Initialize new population by adding the elites
        new_pop = torch.zeros_like(population)
        for i in range(num_elite):
            new_pop[i] = population[sorted_inds[i]]
      
        #The best individual is the one with the best fitness
        best = new_pop[0]
        print('best', pred(model, best.unsqueeze(0), model_type))
        print('worst', pred(model, population[sorted_inds[N - 1]].unsqueeze(0), model_type))
      
        adv_attack = is_attack(model, best + x_orig, t)
      #If not a true adversarial example need to go to next generation
        if adv_attack == False:
            alpha = max(alpha_min, 0.5 * (0.9 ** num_plat))
            rho = max(rho_min, 0.4 * (0.9 ** num_plat))
            # Softmax fitnesses
            soft_fit = F.softmax(-fitness, dim=0) # Negate fitness since we're trying to minimize
            #need to get apply selection and get a new population
            child_pop = selection(population, soft_fit, x_orig, count, alpha, rho, delta_max, num_elite)
            del population
            new_pop[num_elite:] = child_pop
            population = new_pop
            count += 1
            ## Need to retain best fitness
            last_best = best_fit

    return best, adv_attack, count