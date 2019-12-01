# Using AIQNs and Imitation learning to Construct an Optimal Stochastic Policy for Primate Arm Motion


## Objectives

In this project, the objective is to construct an abstract stochastic policy mapping motor cortex states (measured by electrode activation) to some stationary distribution over arm actions. This policy will be learned using collected expert motion data as a reference and feeding this information into a new form of implicit quantile network known as an AIQN which assumes some autoregressive correlation between previous actions and is a function approximator that takes in random noise and outputs a sample from a target distribution. The use of a stationary distribution for action selection is primarily for computational simplicity though it is possible, if not preferable, to lift this approach to a non stationary stochastic policy by using more than one network and acquiring more training data. The hope is to overcome the uncertainty present in arm position and neural activity data through some stochastic policy following some learned distribution (determined via expert data). As this project is simply a proof of concept, the algorithm will restrictively learn in a supervised fashion, this is opposed to the joint supervised and reinforcement learning approach commonly found in imitation learning literature.

## Related Work

### Scalable Muscle-Actuated Human Simulation and Control

This paper is the work that is most similar to what we aim to do in this project. They built a comprehensive musculoskeletal model and control system that was able to reproduce human movements driven by the dynamics of muscle contraction. They took into account variations in the anatomic model to account for movements from the highly typical to the highly stylistic. Using deep reinforcement learning, they delve into the scalable and reliable simulation of anatomical features and movements. Their key contribution was using a scalable, two level imitation learning algorithm. This algorithm was able to deal with the full range of motions in terms of the full-body musculoskeletal model using 346 muscles. They also demonstrated predictive accuracy of motor skills even under varying anatomical conditions ranging from bone deformity, muscle weakness, contracture, and prosthesis use. They also simulate pathological gaits and were able to predictively visualize how orthopedic surgeries would impact the gait of the patients.


## Project Layout

The main components of the Imitation Distributional Policy (IDP) are the IDPAgent model, which is responsible for immitating the actions of the expert agent, and the AIQN which is responsible for constructing a sampling strategy for actions over continuous actions space. These major componentes can be found under the IDP directory. It should be noted (for both users and programmers that the actions returned by the networks are values between [-1, 1]). There are additional directories to house the GAC approach which is to shape rewards to have the agent follow an expert strategy. This method uses the gac_main file for constructing and running the necessary environments.


## Run Commands

To run this project use the command

    python3 -m main

from the main directory.


## Requirements

The current requirements for this project are:
- tensorflow
- docker
- numpy
- gym
- tqdm - for tracking experiment time left


## Docker commands

The main docker commands to be concerned with for this project are as follows.

    docker-compose up --build

Which will construct the docker container and report any logs from said container to the standard output.

    docker-compose up -d

Which will run the docker container in the background.

    docker ps

Will return a list of all running docker containers.

    docker exec -it <image name> bash

And all related commands will allow the user to effectively ssh into the docker container. More useful commands (especially commands to handle cleanup of containers, volumes, and your environment) can be found here: https://docs.docker.com/engine/reference/commandline/docker/.


## Project Tests

To run the tests in the tests file (currently there are only functional tests) use the following command from the home directory.

    python3 -m tests.unit.unit_test_suites


## Current Bugs

There are no current bugs.


## Contributors

Gregory Cho, Olivia Langley, Sean Nathan
