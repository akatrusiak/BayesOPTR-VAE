set term postscript eps color enhanced "Times-Roman" 18 lw 2
set colorsequence classic

#Name and date tag:
labeltext="(c) `whoami`, "
today="`date +%Y/%b/%d`"
set label labeltext.today at screen .01, screen .02

set datafile separator "\t"

#getting the units from the second line of header. I had to resolve to use column numbers instead of using header name...
stat "fort.envelope" every ::1::1 using (sunit=stringcolumn(1)) nooutput
stat "fort.envelope" every ::1::1 using (xunit=(stringcolumn(2))) nooutput
stat "fort.envelope" every ::1::1 using (yunit=(stringcolumn(4))) nooutput

#simulation beam parameters
einit="`head -n3 fort.envelope | tail -n1 | awk '{print $10}'`" #initial energy [MeV]
mass="`head -n1 data.dat | awk '{print $4}'`" #m_0 [MeV/c^2]->MeV (c=1)
charge="`head -n1 data.dat | awk '{print $5}'`" #in integer units of e

set output "optr.eps"
set xlabel "s"."/".sunit
set autoscale xfix
s=10.
sx = 1 
off = 0.0  
plot 'fort.envelope' using 1:(column("x-envelope")) title columnhead(2)."/".xunit with lines lt 1 lc 1, \
'' using 1:(-column("y-envelope")) title "-".columnhead(4)."/".yunit with lines lt 1 lc rgb '#00CC66' , \
'' using 1:(s*column("kx")) title "" with fsteps lt 1 lc rgb '#AAAAAA', \
'' using 1:(-s*column("ky")) title "focal strength" with fsteps lt 1 lc rgb '#AAAAAA',\
'' using 1:(column("FS-x")) title "FS-x" with fsteps lt 1 lc rgb '#FF0000',\
'' using 1:(-column("FS-y")) title "FS-y" with fsteps lt 1 lc rgb '#FF0000',\
'fort.label' using 1:(0):2 with labels rotate by 90 left font "Courier,12" notitle

#olivier - for mac users. 
#It may need some adjustment if anyone uses it on Windows, but we don't have any Windows users at the moment, so we cannot test it
ostype=system("uname -a | awk '{print $1}'")
if (ostype ne "Linux") { system("convert -density 400 optr.eps optr.pdf &")}