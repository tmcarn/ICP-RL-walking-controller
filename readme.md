# Introduction

This is code for the YouTube video on bipeadal walking:
"How to Make a Robot Walk (No AI, Just Physics)"
https://www.youtube.com/watch?v=RXGrTD71FMc

Do not expect this code to be cleanly written or prefectly structured. There is for sure a bunch of spelling mistaces and bad variable naming among other coding crimes. But the code runs and I left it at that. 

To run the code, simply run one of the files called capture_point_... depending on what robot you want to test. I would recomend starting with the capture_point_2d.py. If you understand how the minimal 2d version works, the others are simply an extension of the same principles. The parameters are not perfectly tuned and better performance can definilty be achived. More functions, such as rolling foot contact could be added for faster walking, or a standing controller for the robots with feet.  

Each robot has it's own xml file and an urdf, which yes is redunent, the urdf is only used with pinocchio to get the jacobian of each leg. The velocity controllers in mujoco can get unstable if you use to high gains, then you need to make the simulation step time smaller. 

If you publish any project using this code, it would be nice to have #The5439Workshop in the description, mostly to see if anyone is using it or for what. 

## Dependencies 

I run 
```
python 3.11
mujoco 3.4
numpy 2.3.5
pinocchio 3.8.0
```
and if you want xbox controller, then you also will need pygame. 

