class Logger:
    def __enter__(self):
        print("Log started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Log ended")

    def log(self, message):
        print(f"[LOG] {message}")


with Logger() as logger:
    logger.log("Processing data")
    logger.log("Done")
