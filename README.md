# F1-RL

Final project for [CSE 579: Intelligent Control Through Learning And Optimization](https://courses.cs.washington.edu/courses/cse579/22au/), by [Abhay Deshpande](https://github.com/abhaybd) and [Arnav Thareja](https://github.com/arnavthareja).

This project uses deep reenforcement learning to race 1/10th-scale F1 cars in simulation. The full project report is attached [here](assets/ProjectReport.pdf). The trained agents succesfully learned to race competitively.

Interestingly, the agent was trained on [COTA](https://en.wikipedia.org/wiki/Circuit_of_the_Americas) (left) and was able to transfer to new tracks, such as [Interlagos](https://en.wikipedia.org/wiki/Interlagos_Circuit) (right).

![GIF of agent racing on COTA](assets/cota_highlight.gif)
![GIF of agent racing on Interlagos](assets/saopaulo_highlight.gif)

## Installation instructions

Clone repo, initializing submodules:
```bash
git clone --recurse-submodules https://github.com/abhaybd/F1-RL.git
cd F1-RL
```

Create environment

```bash
conda env create -f env.yaml
conda activate f1-rl
```

Now install f1tenth gym, which has been included as a submodule.

```bash
cd f1tenth_gym
pip install -e .
```

## Usage Instructions

### Train
Train jobs are parameterized by a config file, an example of which is [provided](config/sac.yml). You can train a model using:
```bash
python -m f1rl.train <config_path>
```
Train jobs are logged with [wandb](https://wandb.ai/), so you can set that up or run jobs offline, in which case tensorboard can be used to view the logs.

### Eval
After training a model, you can evaluate models using:
```bash
python -m f1rl.eval <run_path> <checkpoint_name>
```
To download a model that was saved with wandb, `run_path` should be the run path, in the form `entity/project/run`. To run a local model, it should be the path to the run files, and the `--local` flag should be specified. Use the `--help` flag to see a full list of options.

After saving an eval trajectory with the `-s` flag, you can render the trajectory as an image or video. To render it as an image run:
```bash
python -m f1rl.render_recording <recording_path>
```
And to render a video run:
```bash
python -m f1rl.render_recording_vid <recording_path> <vid_save_path>
```
