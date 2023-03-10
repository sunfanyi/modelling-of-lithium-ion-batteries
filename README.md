# modelling-of-lithium-ion-batteries
Imperial College London Mech Eng - Future Clean Transport Technology Project 3

## Fitted Functions for R0, R1 and C1 (keep updating)

### Constant value:

$C_1 = 197.27991163742004$

### Equation 1: First Order Gaussian Function for $R_1 = f(I)$ (not used)

$R_1 = R_1^{0A}exp(-\frac{(I-b)^2}{c}) + d$

Fitted parameters:
- $R_1^{0A} = 0.07292541736130936$
- $b = 0.6251023663197635$
- $c = 1.1847105636748154$
- $d = 0.008692725707500853$

### Equation 2: Arrhenius Equation for $R_0 = f(T)$ (changed 09/03/23)

$R_0 = R_0^{T0}exp(-\frac{E}{R}(\frac{1}{T} - \frac{1}{T_0}))$

where $T$ and $T_0$ are in $K$!!!

Fitted parameters:
- $R_0^{T0} = 0.022130114077306542$
- $E = -18682.80693804173$

Constants:
- $R = 8.314$
- $T_0 = 293.15$

### Equation 3: Arrhenius Equation for $R_1 = f(T)$ (not used)

$R_1 = R_1^{T0}exp(-\frac{E}{R}(\frac{1}{T} - \frac{1}{T_0}))$

Fitted parameters:
- $R_1^{T0} = 0.011824210015710747$
- $E = -16183.500296082062$

### Equation 4: Combined Equation for $R_1 = f(I, T)$ (changed 09/03/23)

$R_1 = R_1^{0A,20^oC}exp(-\frac{(I-b)^2}{c})exp(-\frac{E}{R}(\frac{1}{T}-\frac{1}{T_0})) + d_1exp(-T/d_2) + d_3$

Fitted parameters:
- $R_1^{0A,20^oC} = 0.06381993272009605$
- $b = 0.4983572952818651$
- $c = 0.7350836807623884$
- $E = -5237.390735707343$
- $d_1 = 0.014710912236819855$
- $d_2 = 37.50311325207411$
- $d_3 = 0.00132155691380467$


## First Order ECN Model

Model has been built. They shoule be easily accessible throughout the project. Check the function first_order_ECN() in [tools.py](tools.py) and the example at the last part of [part_2a.ipynb](part_2a.ipynb) for how to use it.

## Some Github Tips

For people feeling uncomfortable with Google Colab and don't have VS Code. This is for Windows only but for Mac it should be similar.

### Step 1 - Check if you have git downloaded

Open Command Prompt (terminal), type git. If there is an error, you need to download git (follow google instruction).

### Step 2 - Clone the repo from github to local

To do this you need to open the terminal and cd to the location where you want to put the folder on. For example, if I want to put it to my Desktop, I type:
```
cd C:\Users\fanyi\Desktop
```
But if you want to switch from C disk to D disk this doesn't work. In this case, do this first:
```
d:
```
![](Figures/cd_help.png)

There is an easier way for cd actually. Open the folder where you want to cd to, simply type cmd on the address bar:

![](Figures/cmd_trick.png)

So after you cd to the location where you want to put the folder on, run:
```
git clone https://github.com/sunfanyi/modelling-of-lithium-ion-batteries.git
```
You may need to login to your github account or something.

Done

### Step 3 - Check

Now you should have access to this github repo. For the following steps always make sure you have cd to the working directory (the folder of modelling-of-lithium-ion-batteries).

Type git log, if you see some commit history, it means you succeeded. Press q to exit.

### Step 4 - Push

After you have made some changes and want to make a commit, run:
```
git add .
```
don't miss the space between add and the dot. Then:
```
git commit -m "Your commit message."
git push
```
Now you should be able to see changes on github. There is a case when you can't push because you didn't pull the stuff, see next step.

### Step 5 - Pull

Whenever you see changes on github, pull it to your local by:
```
git pull
```
Be very careful about this because this will overwrite everything you wrote locally and uncommitted. So make copy before pull if necessary.

If you didin't pull and anyone made a new commit, you can never push. So everytime you want to do the project, go to github and check if there is anything you need to pull.

### Special case:

If two people are working simutaneously without telling each other. People A commit first. When people B want to commit, he won't be able to. He needs to pull it first but everything people B wrote will be overwritten.

In this case, we choose the most stupid method: people B make local copy of his changes, and then pull. Copy and paste his changes to the new version and push again.




