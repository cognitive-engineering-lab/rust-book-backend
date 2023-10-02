import pyro
import pyro.distributions as dist
import torch
from torch import tensor as t
from torch.distributions import constraints
import matplotlib.pyplot as plt
import math
from tqdm.auto import tqdm
import pandas as pd


def normal_params(prefix, shape, init_loc, init_scale, **kwargs):
    loc = pyro.param(f'{prefix}_loc', torch.full(shape, init_loc), **kwargs)
    scale = pyro.param(f'{prefix}_scale', torch.full(shape, init_scale), constraint=constraints.positive)
    return loc, scale

def beta_params(prefix, N):
    a = pyro.param(f'{prefix}_alpha', torch.ones(N), constraint=constraints.positive)
    b = pyro.param(f'{prefix}_beta', torch.ones(N), constraint=constraints.positive)
    return a, b

class IrtModel:
    def __init__(self, subjects_domain, items_domain):
        self.subjects_index = {v: i for i, v in enumerate(subjects_domain)}
        self.num_subjects = len(subjects_domain)

        self.items_index = {v: i for i, v in enumerate(items_domain)}
        self.num_items = len(items_domain)
        
    def model(self, subjects, items, obs):
        with pyro.plate("thetas", self.num_subjects):
            theta = pyro.sample("theta", dist.Normal(0., 1.))
            
        with pyro.plate("item_params", self.num_items):
            diff = pyro.sample("diff", dist.Normal(0., 0.1))
            disc = pyro.sample("disc", dist.Normal(0., 0.1))
            lambdas = pyro.sample("lambdas", dist.Beta(1., 1.))

        with pyro.plate("observed", obs.size(0)):
            probs = lambdas[items] + (1. - lambdas[items]) * torch.sigmoid(disc[items] * (theta[subjects] - diff[items]))
            pyro.sample("obs", dist.Bernoulli(probs=probs), obs=obs)
        
    def guide(self, *args):
        with pyro.plate("thetas", self.num_subjects):
            pyro.sample("theta", dist.Normal(*normal_params("theta", self.num_subjects, 0., 1.)))
            
        with pyro.plate("item_params", self.num_items):
            pyro.sample("diff", dist.Normal(*normal_params("diff", self.num_items, 0., 1.0e1)))
            pyro.sample("disc", dist.Normal(*normal_params("disc", self.num_items, 1., 1.0e-6, constraint=constraints.positive)))
            pyro.sample("lambdas", dist.Delta(pyro.param('lambdas_est', torch.full((self.num_items,), 0.5), constraint=constraints.interval(0., 1.))))

    def train(self, subjects, items, obs, epochs=500):      
        pyro.clear_param_store()

        adam = pyro.optim.Adam({"lr": 0.1})
        elbo = pyro.infer.Trace_ELBO()
        svi = pyro.infer.SVI(self.model, self.guide, adam, elbo)

        losses = []
        for _ in tqdm(range(epochs)):
            loss = svi.step(subjects, items, obs)
            losses.append(loss)

        return losses

def train(answers, q_df, **kwargs):
    answers = answers[~answers.questionId.isna()]
    subjects_domain = answers.sessionId.unique().tolist()
    items_domain = answers.questionId.unique().tolist()

    model = IrtModel(subjects_domain, items_domain)

    subjects = t([model.subjects_index[s] for s in answers.sessionId])
    items = t([model.items_index[i] for i in answers.questionId])
    obs = t(answers.correct.tolist()).float()

    _losses = model.train(subjects, items, obs, **kwargs)

    # plt.figure(figsize=(5, 2))
    # plt.plot(losses)
    # plt.xlabel("SVI step")
    # plt.ylabel("ELBO loss")

    item_acc = answers.groupby('questionId').correct.mean()
    item_counts = answers.groupby('questionId').size()

    irt_df = pd.DataFrame([{
        'id': id,
        'disc': pyro.param("disc_loc")[model.items_index[id]].item(),
        'diff': pyro.param("diff_loc")[model.items_index[id]].item(),
        'lambda': pyro.param("lambdas_est")[model.items_index[id]].item(),
        "acc": item_acc[id],
        "N": item_counts[id],
        "r": q_df[q_df.id == id].r.iloc[0]
    } for id in items_domain if id in q_df.id.tolist()])

    subj_acc = answers.groupby('sessionId').correct.mean()
    subj_counts = answers.groupby('sessionId').size()

    irt_subj_df = pd.DataFrame([{
        "id": id,
        "theta": pyro.param("theta_loc")[model.subjects_index[id]].item(),
        "acc": subj_acc[id],
        "N": subj_counts[id],
    } for id in subjects_domain])

    return irt_df, irt_subj_df


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def plt_item(id):
    print(lambdas[id], disc[id], diff[id])
    f_3pl = lambda theta: lambdas[id] + (1. - lambdas[id]) * sigmoid(disc[id] * (theta - diff[id]))
    f_2pl = lambda theta: sigmoid(disc[id] * (theta - diff[id]))
    
    f = f_3pl
    xs = np.arange(-3., 3., 0.1)
    plt.plot(xs, [f(x) for x in xs])
    ax = plt.gca()
    ax.set_ylim(0., 1.)
    ax.set_xlabel('Ability')
    ax.set_ylabel('P(correct)')

class MultiIrtModel:
    def __init__(self, subjects_domain, items_domain, hidden_size):
        self.hidden_size = hidden_size
        self.subjects_index = {v: i for i, v in enumerate(subjects_domain)}
        self.num_subjects = len(subjects_domain)

        self.items_index = {v: i for i, v in enumerate(items_domain)}
        self.num_items = len(items_domain)
        
    def model(self, subjects, items, obs):
        hidden_plate = pyro.plate("hidden", self.hidden_size)
        subj_plate = pyro.plate("thetas", self.num_subjects)
        item_plate = pyro.plate("item_params", self.num_items)

        with subj_plate, hidden_plate:
            theta = pyro.sample("theta", dist.Normal(0., 1.))

        with item_plate:
            with hidden_plate:
                a = pyro.sample("a", dist.Normal(0., 0.1))
            d = pyro.sample("d", dist.Normal(0., 0.1))

        with pyro.plate("observed", obs.size(0)):
            logits = torch.sum(a[:,items] * theta[:,subjects], dim=0) + d[items]
            pyro.sample("obs", dist.Bernoulli(logits=logits), obs=obs)
        
    def guide(self, *args):
        hidden_plate = pyro.plate("hidden", self.hidden_size)
        subj_plate = pyro.plate("thetas", self.num_subjects)
        item_plate = pyro.plate("item_params", self.num_items)

        with subj_plate, hidden_plate:
            pyro.sample("theta", dist.Normal(*normal_params("theta", (self.hidden_size, self.num_subjects), 0., 1.)))
            
        with item_plate:
            with hidden_plate:
                pyro.sample("a", dist.Normal(*normal_params("a", (self.hidden_size, self.num_items), 0., 1.0e-6)))
            pyro.sample("d", dist.Normal(*normal_params("d", (self.num_items,), 1., 1.0e1)))            

    def train(self, subjects, items, obs, epochs=500):      
        pyro.clear_param_store()

        adam = pyro.optim.Adam({"lr": 0.1})
        elbo = pyro.infer.Trace_ELBO()
        svi = pyro.infer.SVI(self.model, self.guide, adam, elbo)

        losses = []
        for _ in tqdm(range(epochs)):
            loss = svi.step(subjects, items, obs)
            losses.append(loss)

        return losses

def train_multi(answers, q_df, hidden_size, **kwargs):
    answers = answers[~answers.questionId.isna()]
    subjects_domain = answers.sessionId.unique().tolist()
    items_domain = answers.questionId.unique().tolist()

    model = MultiIrtModel(subjects_domain, items_domain, hidden_size)

    subjects = t([model.subjects_index[s] for s in answers.sessionId])
    items = t([model.items_index[i] for i in answers.questionId])
    obs = t(answers.correct.tolist()).float()

    losses = model.train(subjects, items, obs, **kwargs)

    plt.figure(figsize=(5, 2))
    plt.plot(losses)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")

    item_acc = answers.groupby('questionId').correct.mean()
    item_counts = answers.groupby('questionId').size()

    irt_df = pd.DataFrame([{
        'id': id,
        'a': pyro.param("a_loc")[:,model.items_index[id]].detach().numpy(),
        'd': pyro.param("d_loc")[model.items_index[id]].item(),
        "acc": item_acc[id],
        "N": item_counts[id],
        "r": q_df[q_df.id == id].r.iloc[0]
    } for id in items_domain if id in q_df.id.tolist()])
    irt_df['a_mean'] = irt_df['a'].map(lambda a: a.mean())    

    subj_acc = answers.groupby('sessionId').correct.mean()
    subj_counts = answers.groupby('sessionId').size()

    irt_subj_df = pd.DataFrame([{
        "id": id,
        "theta": pyro.param("theta_loc")[:,model.subjects_index[id]].detach().numpy(),
        "acc": subj_acc[id],
        "N": subj_counts[id],
    } for id in subjects_domain])
    irt_subj_df['theta_mean'] = irt_subj_df['theta'].map(lambda a: a.mean())    

    return irt_df, irt_subj_df