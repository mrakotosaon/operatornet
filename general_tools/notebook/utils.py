'''
Created on Apr 27, 2017

@author: optas
'''
import subprocess


def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")


def mnemonics():
    s = '1. Start interactive matplotlib plots: %matplotlib nbagg\n'
    s += '2. Use plotly: from plotly.offline import init_notebook_mode\n init_notebook_mode(connected=True)\n'
    s += '3. Turn off plotting. plt.ioff() and plt.close() in for-loop.'
    print s