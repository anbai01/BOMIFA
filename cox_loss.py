import torch
def loss(y_pred,label):
    hazard_ratio = torch.exp(y_pred)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
    uncensored_likelihood = y_pred.T - log_risk
    censored_likelihood = uncensored_likelihood * label
    neg_likelihood_ = -(torch.sum(censored_likelihood))
    num_observed_events = torch.tensor(1.0)
    neg_likelihood = neg_likelihood_ / num_observed_events

    return neg_likelihood