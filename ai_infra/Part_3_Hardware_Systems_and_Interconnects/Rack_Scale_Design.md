# Rack-Scale Design and Power/Thermal Math

A rigorous engineering analysis of the physical, electrical, and thermal constraints governing 2026 AI infrastructure. The transition from $10$ kW general-purpose compute racks to $150$ kW NVL72 and Helios racks invalidates legacy data center design. We present the mathematical models driving power distribution (48V busbars) and direct-to-chip liquid cooling.

**Prerequisites**: [GPU_Architecture](../Part_2_Microarchitecture/GPU_Architecture.md), [Networking_and_Interconnect](Networking_and_Interconnect.md).

---

## 1. Electrical Power Distribution

### 1.1 The Shift to 48V Distribution
Traditional racks distribute $12$ Volts (V) DC to motherboards. As rack power $P$ scales to $120$ kW, utilizing 12V necessitates immense current ($I = P/V = 10,000$ Amps).
- **The $I^2R$ Loss Math**: Power loss due to wire resistance $R$ scales quadratically with current.
$$P_{loss} = I^2 R$$
- Moving from 12V to 48V distribution reduces current by a factor of 4. Consequently, $I^2R$ losses are reduced by a factor of 16 ($4^2$). This is mathematically mandatory to prevent busbars from melting and to reduce the physical cross-sectional area of copper routing required in the rack.

### 1.2 Point-of-Load (PoL) Conversion
At the GPU package, the 48V must be stepped down to the core voltage ($\sim 0.7$V).
- The PoL step-down ratio is $\approx 68:1$.
- If a GPU consumes $1000$ W at $0.7$ V, it draws $\approx 1428$ Amps. Delivering $>1000$ Amps requires placing Voltage Regulator Modules (VRMs) millimeters away from the silicon die to minimize PCB trace impedance $Z$, avoiding excessive $I \cdot Z$ voltage droop during nanosecond-scale current transients (di/dt).

---

## 2. Thermal Thermodynamics

### 2.1 The Sensible Heat Equation
A $120$ kW rack ($120,000$ J/s) converts $\approx 99.9\%$ of its electrical power into heat. Air cooling fundamentally fails beyond $\approx 40$ kW/rack due to the low specific heat capacity and low density of air.

We apply the sensible heat equation for Direct-to-Chip (D2C) liquid cooling:
$$Q = \dot{m} C_p \Delta T$$
Where:
- $Q = 120,000$ W
- $C_p \approx 4184$ J/(kg·K) (for water-based coolant)
- $\Delta T$ is the acceptable coolant temperature rise (typically restricted to $10^\circ$ C to prevent HBM from crossing its $95^\circ$ C thermal limit, which would trigger extreme bandwidth throttling).

Solving for mass flow rate $\dot{m}$:
$$\dot{m} = \frac{120,000}{4184 \times 10} \approx 2.87 \text{ kg/s}$$
This requires $\approx 45$ Gallons Per Minute (GPM) of continuous flow per rack, necessitating facility-level Coolant Distribution Units (CDUs) and massive multi-megawatt external dry coolers.

### 2.2 Two-Phase Boiling (Evaporative Cooling)
To support $200$ kW racks (e.g., future NVL144), engineers must rely on the latent heat of vaporization $h_{fg}$:
$$Q = \dot{m}_{evap} h_{fg} + \dot{m} C_p \Delta T$$
Because $h_{fg}$ is astronomically higher than $C_p$, two-phase cooling requires orders of magnitude less fluid flow, but introduces severe mechanical complexities managing vapor pressure dynamics.

---

## 3. Physical Layout Constraints

### 3.1 The NVL72 Copper Backbone
The NVL72 rack connects 72 GPUs to 9 NVSwitch trays using a dense copper backplane, rather than optical cables.
- **Latency and Cost**: Copper is significantly cheaper and lower-latency than active optical transceivers.
- **Trace Length Math**: However, at $224$ Gbps PAM4 signaling, the Nyquist frequency attenuation strictly limits passive copper traces to $\le 1$ meter.
This physically dictates the rack layout: NVSwitch trays *must* be positioned in the exact physical center of the rack (the "spine"), with GPU compute trays positioned symmetrically above and below, guaranteeing that no physical trace exceeds the $1$ meter constraint.

---

## 4. Further Reading

- [Blackwell_Architecture](../Part_2_Microarchitecture/Blackwell_Architecture.md) - Examining the $1200$ W TDP of B300.
- [Networking_and_Interconnect](Networking_and_Interconnect.md) - The signaling frequencies forcing dense rack-scale consolidation.