I'm traning on d4rl ant maze, and I wanna fix one starting point
(lower left corner) and then target three points, upper right corner,
upper left corner, lower right corner...

To start, we can just pass in empty test-time goal pickles, to get
a pretrained solver to use for the rest of the experiment (say 3 pretrained)
solvers, one for each seed (0, 1, 2) and just make sure to keep the seeds
consistent at test time



