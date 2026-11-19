# Autonomous Field Mode: physical switch design

Status: **preference storage and Settings UI are implemented**. The GPIO/I2C
hardware agent, LED watchdog, boot-time auto-start, and live recording gate are
not yet implemented. The Preferences API must not be treated as proof that
the physical switch is connected.

## Hardware prototype (Jetson Orin Nano developer kit)

Use a 3.3 V MCP23017 I2C GPIO expander for three or four low-current LEDs
and a latching SPST switch. The Orin Nano reference carrier board's GPIO
outputs may only drive weak loads; **do not connect status LEDs directly to
the Jetson GPIO pins**. The Jetson I2C header (3.3 V, ground, SDA, SCL)
can control the expander when configured and verified. Wire only with the
Jetson fully powered down and disconnected.

Suggested logical MCP23017 pins (not Jetson 40-pin header numbering):

- GPA0: red LED, with a series current-limiting resistor.
- GPA1: yellow LED, with a series current-limiting resistor.
- GPA2: green LED, with a series current-limiting resistor.
- GPA3: optional blue logging LED, with a series resistor.
- GPA4: latching SPST switch input, internal pull-up enabled; closed
  connects the input to MCP23017 ground.

Use basic low-current LEDs and start with approximately 1 kΩ series resistors
from each output. Confirm LED forward voltage, polarity, visibility, total
expander load, and board breakout wiring before field use. Power the
MCP23017 from 3.3 V, never 5 V I2C signaling into Jetson pins.

## Switch modes

The API offers a persisted `mode`:

- `inference_and_logging` (default): switch OFF = idle/ready; switch ON =
  create and start an acquisition session with both inference and logging;
  switch OFF = stop safely and close the session.
- `logging_only`: switch OFF = inference continues but detections are not
  persisted; switch ON = begin recording detections, and switch OFF =
  finish the current logging session without stopping inference.

The hardware controller must fetch preferences from
`GET /api/field-switch/config` on start and before each **new** acquisition
session. `PUT /api/field-switch/config` validates and atomically persists
the user's selection in the existing SQLite metadata table. A newly saved
preference must not retroactively alter an active session. If a switch is
already physically ON when the service starts, explicitly reconcile its
level after device discovery and model/camera readiness.

A logging-only hardware agent will need a separate per-frame persistence
gate: existing sessions currently freeze `record_data` at creation.
Do not implement logging-only mode by silently stopping and restarting a
running inference session. Ensure that mode and recording gate state are
audited against the session and never report logging success until the
database write is verified.

## Health indicators

- Red solid: unrecoverable error; red flashing: recovery in progress.
- Yellow: loading service/model/camera.
- Green solid: model loaded and ready.
- Green blinking: fresh inference-frame heartbeat.
- Blue optional: confirmed detection records are being written.

A separate watchdog should drive the LED status so a frozen application
can turn the LED red. On loss of the MCP23017, recording must remain
available via normal application controls, and the hardware agent must
expose a noncritical hardware fault.

Before enabling field deployment, test switch debounce and boot ON state,
restarts, hot/unplugged camera, SQLite disk-full behavior, safe session
shutdown, and two modes with synthetic source/camera inputs. Confirm
systemd behavior on the actual Jetson/JetPack image.
