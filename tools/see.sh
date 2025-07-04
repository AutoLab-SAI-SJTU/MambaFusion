ps -eo pid,comm --sort=-vsz | awk 'NR==1 {print "Swap (GB)\tPID\tCommand"} NR>1 {print $1}' | while read pid; do
  if [ -f /proc/$pid/status ]; then
    swap=$(grep VmSwap /proc/$pid/status | awk '{ print $2 }')
    if [ -n "$swap" ]; then
      swap_gb=$(echo "scale=2; $swap/1024/1024" | bc)
      if (( $(echo "$swap_gb > 0" | bc -l) )); then
        cmd=$(ps -p $pid -o comm=)
        echo -e "${swap_gb}\t\t${pid}\t${cmd}"
      fi
    fi
  fi
done | sort -nr | head -n 5
