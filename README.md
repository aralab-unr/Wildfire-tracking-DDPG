# Multi-UAV Collaborative Deep Learning and Consensus to Maximize Coverage Over Continuous and Dynamic Wildfire Environment

This is the implementation of methods discussed in the paper, proposed in IROS 2023. To access the paper used this [link](https://drive.google.com/file/d/1pgpPX0bzjf6ChFjhr3SZ44L6XKJNPrqr/view?usp=sharing). You are free to use all or part of the codes here presented for any purpose, provided that the paper is properly cited and the original authors properly credited. All the files here shared come with no warranties.


This project was built on Python 3.9.13 and PyTorch 1.13.0. To access the results, (Jupyter Notebook)[http://jupyter.readthedocs.io/en/latest/install.html] is required.

## Files
* Method1: Related to Independent UAVs Without Any Communication
* Method2: Related to Independent UAVs With Memory Sharing Only
* Method3: Related to Dependent Agents With Positional And Memory Sharing
* Method4: Related to Consensus-Based Agent Network
* Code1-4.png: Images of Method4 Timelapse

## Abstract
We all are aware of consequences of the wildfire, especially when controlling the spread is challenging. Firefighters try to understand the environment in planning strategies to get the wildfire under control. To analyze the environment, gathering information is crucial. People around the world came up with their way of approaches to achieve this task. Deploying UAVs are one of the best ways for tracking, covering, and gathering information about wildfire. A team of UAVs has additional advantages too. There were many approaches discussed regarding the use of multiple UAVs to track and cover the fire region. Very few were discussed about the communication between the UAVs with respect to their performance. In this paper, we discuss 4 different variations of communications within the team of UAVs and compare them based on a set of performance measures like not just the reward collection but also the coverage, the duration of tracking, and other environmental metrics.

## How to use
The project is divided in 2 parts: Learning and Results. </ br>
In Learning part, The models in their respective folders will be trained and the data to plot results is generated. </ br>
1. To train Method1, in `Terminal`, go to Method1 folder and type 
``` python main_ddpg.py ```
2. To train Method2, in `Terminal`, go to Method2 folder and type 
``` python main_ddpg.py ```
3. To train Method3, in `Terminal`, go to Method3 folder and type 
``` python main_ddpg.py ```
4. To train Method4, in `Terminal`, go to Method4 folder and type 
``` python main_ddpg.py ```


In Results part, The generated data is used to plot graphs for. </ br>
1. Score History
2. Coverage Ratio
3. Fastest Coverage
4. Fastest Tracking
5. Fire Fallout
6. Environment Fallout

## Contact
For any questions regarding this project, please contact through any of these emails, gauravsrikar@nevada.unr.edu or aralab2018@gmail.com .
