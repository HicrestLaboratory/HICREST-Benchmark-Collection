## To ensure these script work from compute nodes

```bash
# On a MonteCimone peak node
python3.10 -m venv .peak-venv
pip3.10 install apache-iotdb

# On a MonteCimone blade node
python3.11 -m venv .blade-venv
pip3.11 install apache-iotdb

# Then make sure that your scripts:
# source ~/.<peak|blade>-venv/bin/activate
```