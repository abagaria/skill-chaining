This is an experiment to figure out lambda's dependence on num_steps.

We're hoping to get a curve for each lambda relating num_steps to things like volume taken up by latent space points.

We want each point in latent space to have roughly the same volume (?), so we want to see how we need to change
lambda with num_steps to keep that quantity constant. We'll plot this point as well.

This requires really understanding the range of lambdas. So, we're going to be setting scaling_method to none,
so that we can isolate the effects of changing lambda. We'll have something like 6 num_steps, and 6 lambdas.

We'll make a script that runs on Grid, that we can specify a set of hyperparameters with.