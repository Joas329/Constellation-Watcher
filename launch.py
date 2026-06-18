#!/usr/bin/env python3

# Import necessary modules like this:
# from daemon.daemon import ConstellationDaemon
# from webapp.app import create_app


def main():
    daemon = ConstellationDaemon()
    daemon.start()

    app = create_app()
    app.run(host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()