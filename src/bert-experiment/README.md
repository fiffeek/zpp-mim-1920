# bert pretraining module
## run
if you run the first time:
```./commands.sh```
afterwards just go with:
```sudo python3 run_bert.py```

## docker tests
```
docker build -t zpp .
docker run -ti -v $HOME/.config/gcloud:/root/.config/gcloud zpp
```