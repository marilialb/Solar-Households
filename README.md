# Solar Households

**An agent-based simulation of a small solar-powered neighbourhood, where every household learns to manage its own energy.**

**Live app:** [solarhouseholds.org](https://solarhouseholds.org) · 📝 **Research survey:** [Give feedback](https://forms.office.com/r/FcG6dysQ3B) · 📄 [Participant Information Statement (PDF)](https://solarhouseholds.org/pis.pdf)

---

## Overview

Solar Households simulates a virtual neighbourhood of houses, apartments and small businesses. Each one may or may not have solar panels, a battery, or an electric car. As the simulation runs, you can watch each household manage its energy, earn or spend money, and gradually learn to do better.

This project is part of a research study. See [Research & Feedback](#research--feedback) below.

## How It Works

The simulation is built around three main components.

### TinyML — the household's brain

Each household has a small AI model that observes six inputs:

| Input | Description |
|---|---|
| Battery level | How full the battery is |
| Solar generation | How much sun is shining |
| Time of day | Current time in the simulated day |
| Budget | How much money is left |
| Car charge | How charged the electric vehicle is |
| Power demand | How much power the household needs |

It outputs a single value between **0 and 1**, a dial from *conserve energy* (0) to *use energy freely* (1).

The model learns through trial and error. After every tick it receives a **reward** based on whether it used solar power wisely, wasted any, or left the house short of energy, then adjusts its internal parameters to improve. There is no teacher and no manual.

### Household — the simulated house

Each household is assigned one of six personality types, from **Solar Neglect** (barely uses solar) through to a type that maximises solar efficiency. Types influence:

- Income
- Battery size
- Solar panel capacity
- Electric car ownership

On every **tick**, a household:

1. Receives sunlight (or doesn't)
2. Lets its brain decide how much energy to use
3. Charges the car if needed
4. Gains or loses money
5. Learns from the outcome

If a household runs out of budget it drops out of the simulation, though neighbours can bring it back. The simulation runs for a fixed period until all households have exhausted their budgets.

### Grid — the neighbourhood

The Grid arranges households in a configurable grid of rows and columns and advances all of them simultaneously each tick. It also runs a **local energy market**: a household with a low battery can buy energy from a neighbour with a full one at the current market price, which rises and falls with overall scarcity across the neighbourhood.

## Getting Started

>  Update the commands below to match your project's actual setup.

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
<install command, e.g. npm install / pip install -r requirements.txt>

# Run the app
<run command, e.g. npm start / python main.py>
```

Or simply try the hosted version at **[solarhouseholds.org/app](https://solarhouseholds.org/app)**.

## Project Structure

> Replace with your actual layout.

```
.
├── src/
│   ├── tinyml/        # Household learning model
│   ├── household/     # Household simulation logic
│   └── grid/          # Neighbourhood grid and energy market
├── public/
├── README.md
└── LICENSE
```

## Configuration

The grid size (rows × columns) and household settings can be edited in the app. Document any additional config options, environment variables, or parameters here.

## Research & Feedback

This application is part of a university research project. If you've explored the simulation, please:

1. Read the [Participant Information Statement](https://solarhouseholds.org/pis.pdf)
2. Complete the [short survey](https://forms.office.com/r/FcG6dysQ3B) (under 2 minutes)

Your feedback directly informs the next stage of the project.

### Research Team

- Dr Marilia Bergamo
- Associate Professor Craig Hight
- Dr Andrea Cassin
- Dr Richard Wood

University of Newcastle, Australia. Contact details are available in the Participant Information Statement.

## Contributing

Contributions, issues and feature requests are welcome. Feel free to open an [issue](../../issues) or submit a pull request.

## License

MIT
