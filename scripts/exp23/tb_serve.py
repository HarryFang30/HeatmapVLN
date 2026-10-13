"""TensorBoard with its built-in plugins only (this env's add-on plugins register graph_ascend twice)."""
import sys
from tensorboard import default, program
addons = set(default.get_dynamic_plugins())
tb = program.TensorBoard(plugins=[p for p in default.get_plugins() if p not in addons])
tb.configure(argv=[sys.argv[0], *sys.argv[1:]])
sys.exit(tb.main())
