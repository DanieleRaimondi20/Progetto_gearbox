from datetime import datetime
from pathlib import Path
from logging import getLogger
from progetto_gearbox.logging.logger_configuration import setup_logger
from bokeh.server.server import Server
# from progetto_gearbox.demos.gear_demo import main
from progetto_gearbox.demos.gearbox_demo import main



timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = Path("logs") / f"{Path(__file__).stem}_{timestamp}.log"
setup_logger(str(log_file))
logger = getLogger(__name__)

def bkapp(doc):
    main(doc=doc, server=server)

server = Server({'/': bkapp})
server.start()

server.io_loop.add_callback(server.show, "/")
server.io_loop.start()