FROM europe-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest

WORKDIR /

COPY trainer /trainer

ENTRYPOINT [ "python", "-m", "trainer.task" ]