#!/bin/bash
  
node=$(hostname -s)
ITXT=$node.perf.pid
nvpid=$(< $ITXT)
kill $nvpid
#kill -9 $nvpid
echo "killed process $nvpid on $node"

exit

