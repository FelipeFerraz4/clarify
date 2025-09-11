import logging
from pygls.server import LanguageServer
from pygls.protocol import default_converter
from lsprotocol.types import TEXT_DOCUMENT_DID_OPEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = LanguageServer("clarify-lsp", "0.1.0")


@server.feature(TEXT_DOCUMENT_DID_OPEN)
def did_open(params):
    logger.info(f"Document opened: {params.text_document.uri}")
    server.show_message("Hello, world! Editing file: " + params.text_document.uri)


def main():
    logger.info("Starting Clarify LSP server...")
    server.start_io()


if __name__ == "__main__":
    main()
