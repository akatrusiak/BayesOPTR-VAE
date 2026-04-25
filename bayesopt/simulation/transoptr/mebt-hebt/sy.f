      SUBROUTINE TSYSTEM
      COMMON/SCPARM/QSC,ISC,CMPS
      COMMON/MOM/P,BRHO,pMASS,ENERGK,GSQ,ENERGKi,charge,current
      COMMON/PRINT/IPRINT
      COMMON/SS/SX(13,6)
      COMMON/BLOC1/SGX1,SGY1,SGX2,SGY2,AGX1,AGY1,QM1,AGX2,AGY2,QM2,YCB1,
     &ABX1,ABY1,ABZ1,EMBX1,RBX1,ABX2,ABY2,ABZ2,EMBX2,RBX2,YCB2,AGX3,AGY3
     &,QM3,AGX4,AGY4,QM4,RFA1,RFA2,RFP1,XCB1,YCB3,AGX5,AGY5,QM5,AGX6,AGY
     &6,QM6,YCB4,AGX7,AGY7,QM7,XCB2,AGX8,AGY8,QM8,RFA3,RFA4,RFP2,AGX9,AG
     &Y9,QM9,AGX10,AGY10,QM10,XCB3,YCB5,AGX11,AGY11,QM11,RFA5,RFA6,RFP3,
     &RFA7,RFA8,RFP4,AGX12,AGY12,QM12,AGX13,AGY13,QM13,XCB4,YCB6,AGX14,A
     &GY14,QM14,RFA9,RFA10,RFP5,RFA11,RFA12,RFP6,AGX15,AGY15,QM15,AGX16,
     &AGY16,QM16,XCB5,YCB7,AGX17,AGY17,QM17,RFA13,RFA14,RFP7,RFA15,RFA16
     &,RFP8,AGX18,AGY18,QM18,AGX19,AGY19,QM19,XCB6,YCB8,AGX20,AGY20,QM20
     &,RFA17,RFA18,RFP9,XCB7,YCB9,AGX21,AGY21,QM21,AGX22,AGY22,QM22,XCB8
     &,YCB10,AGX23,AGY23,QM23,AGX24,AGY24,QM24

      CMPS=3.69 ! Number of cm per step, for plotting only
      wo=1.0 ! Weight aberration from optical elements

      ! Built from acc at commit: 142dba29, commit date: 2024-05-27

      call genshift(SGX1,SGX2,SGY1,SGY2)
      call marker('MEBT:RPM5')
      call drift(2.0041,".")
      ! mcat-mebt-set-chopslit
      bfoc=bx1/10.
      ! MCAT-verb: mebt-set-chopslit-spot
      !nlines: 3
      !call twissmatch(1, 0.0, bfoc, 1., 1)
      !call twissmatch(3, 0.0, bfoc, 1., 1)
      !  return
      ! MCAT-verb: mebt-set-timefocus
      !nlines: 4
      !energerr=energk-0.153*qmass
      !call fitarb(0.0,energerr,5.,1)
      !call fit(1,5,5,0.0,1.,1)
      !  return
      ! MCAT-verb: mebt-rotator-debunch
      !nlines: 4
      !energerr=energk-0.153*qmass
      !call fitarb(0.0,energerr,5.,1)
      !call fit(1,6,6,0.0,1.,1)
      !  return
      ! mebt-frame-rotation
      call RT(45.0)
      ! MCAT Mass Specification (amu)
      qmass=pmass/931.595
      call twissfind(1,ax1,bx1)
      call twissfind(3,ay1,by1)
      ! MEBT:CHOPSLIT
      call drift(16.1519,".")
      ! Element of unknown type: strp
      call drift(14.3078,".")
      ! MEBT:IV6
      call drift(11.8,".")

      call genshift(AGX1,0.0,AGY1,0.0)
      ! Magnetic quadrupole MEBT:Q6
      call mquad(None,2.6,18.0,wo,'MEBT:Q6')
      call genshift(-AGX1,0.0,-AGY1,0.0)

      call drift(12.0076,".")

      call genshift(AGX2,0.0,AGY2,0.0)
      ! Magnetic quadrupole MEBT:Q7
      call mquad(None,2.6,18.0,wo,'MEBT:Q7')
      call genshift(-AGX2,0.0,-AGY2,0.0)

      call drift(11.9334,".")
      ! Steerer MEBT:YCB7A
      call genshift(0.,0.,0.,YCB1)
      call drift(24.0675,".")

      call genshift(ABX1,0.0,ABY1,0.0)
      call edge(22.5,29.9975,45.0,0.0,0.55,0.0,5.72,0.0,wo)
      call bend(29.9975,45.0/2,0.0,'.')
      call marker('MEBT:MB1')
      call genshift(0.,EMBX1,0.,RBX1)
      call bend(29.9975,45.0/2,0.0,'.')
      call edge(22.5,29.9975,45.0,0.0,0.55,0.0,5.72,0.0,wo)
      call genshift(-
     &ABX1*cos(0.785398*pi/180)+ABZ1*cos(0.785398*pi/180),0.0,-ABY1,0.0)
      call drift(-ABX1*sin(0.785398*pi/180),".")

      call drift(24.8355,".")
      call marker('MEBT:RPM7')
      ! mcat-corner-single-achromat-1
      ! MCAT-verb: mebt-corner-single-achromat
      !nlines: 2
      !call twissmatch(1,0.0,bx1,1.,1)
      !call fit(1,4,3,0.0,1.,1)
      call drift(5.1677,".")
      ! MEBT:XSLIT7
      call marker('MEBT:FC7')
      call drift(29.9988,".")

      call genshift(ABX2,0.0,ABY2,0.0)
      call edge(22.5,29.9975,45.0,0.0,0.55,0.0,5.72,0.0,wo)
      call bend(29.9975,45.0/2,0.0,'.')
      call marker('MEBT:MB2')
      call genshift(0.,EMBX2,0.,RBX2)
      call bend(29.9975,45.0/2,0.0,'.')
      call edge(22.5,29.9975,45.0,0.0,0.55,0.0,5.72,0.0,wo)
      call genshift(-
     &ABX2*cos(0.785398*pi/180)+ABZ2*cos(0.785398*pi/180),0.0,-ABY2,0.0)
      call drift(-ABX2*sin(0.785398*pi/180),".")

      call drift(24.5577,".")
      ! Steerer MEBT:YCB7B
      call genshift(0.,0.,0.,YCB2)
      call drift(11.4437,".")

      call genshift(AGX3,0.0,AGY3,0.0)
      ! Magnetic quadrupole MEBT:Q8
      call mquad(None,2.6,18.0,wo,'MEBT:Q8')
      call genshift(-AGX3,0.0,-AGY3,0.0)

      call drift(12.005,".")

      call genshift(AGX4,0.0,AGY4,0.0)
      ! Magnetic quadrupole MEBT:Q9
      call mquad(None,2.6,18.0,wo,'MEBT:Q9')
      call genshift(-AGX4,0.0,-AGY4,0.0)

      call drift(12.9324,".")
      call marker('MEBT:LPM9')
      ! Element of unknown type: sid
      call marker('MEBT:FC9')
      call drift(2.8829,".")
      ! mcat-mebt-corner-single-achromat-2
      ! MCAT-verb: mebt-corner-single-achromat
      !nlines: 6
      !call dr(9.96,'half-rebuncher')
      !call twissmatch(1, 0.0, bx1, 1., 1)
      !call twissmatch(3, 0.0, bx1, 1., 1)
      !call fit(1,1,6,0.0,1.,1)
      !call fit(1,2,5,0.0,1.,1)
      !  return
      call drift(5e-05,".")
      ! RF cavity/linac: ISAC1:MEBT
      call linacn(100,992,None,19.926,3.536e+07,None,'ISAC1:MEBT')
      call drift(3.09995,".")
      ! mcat-set-mebt-rebuncher
      ! MCAT-verb: set-mebt-rebuncher
      !nlines: 6
      !call dr(174.84,'mid-tank1')
      !call fit(1,5,5,0.0,1.,1)
      !call fit(1,6,5,0.0,1.,1)
      !energerr=energk-0.15300*qmass
      !call fitarb(0.0,energerr,10.,1)
      !  return
      call drift(3.8485,".")
      ! Steerer MEBT:XCB9
      call genshift(0.,XCB1,0.,0.)
      ! Steerer MEBT:YCB9
      call genshift(0.,0.,0.,YCB3)
      call drift(10.8813,".")

      call genshift(AGX5,0.0,AGY5,0.0)
      ! Magnetic quadrupole MEBT:Q10
      call mquad(None,2.6,18.0,wo,'MEBT:Q10')
      call genshift(-AGX5,0.0,-AGY5,0.0)

      call drift(11.7688,".")

      call genshift(AGX6,0.0,AGY6,0.0)
      ! Magnetic quadrupole MEBT:Q11
      call mquad(None,2.6,18.0,wo,'MEBT:Q11')
      call genshift(-AGX6,0.0,-AGY6,0.0)

      call drift(5.8844,".")
      ! Steerer MEBT:YCB11
      call genshift(0.,0.,0.,YCB4)
      call drift(5.8844,".")

      call genshift(AGX7,0.0,AGY7,0.0)
      ! Magnetic quadrupole MEBT:Q12
      call mquad(None,2.6,18.0,wo,'MEBT:Q12')
      call genshift(-AGX7,0.0,-AGY7,0.0)

      call drift(5.8844,".")
      ! Steerer MEBT:XCB12
      call genshift(0.,XCB2,0.,0.)
      call drift(5.8844,".")

      call genshift(AGX8,0.0,AGY8,0.0)
      ! Magnetic quadrupole MEBT:Q13
      call mquad(None,2.6,18.0,wo,'MEBT:Q13')
      call genshift(-AGX8,0.0,-AGY8,0.0)

      call drift(18.7993,".")
      ! MEBT:IV13
      call drift(4.572,".")
      ! dtl-xy-match
      !DTL tank-1 entrance match (midpoint focus) - LORASR checked.
      !energerr=energki-energk
      !call FITARB(0D0,energerr,1.,1)
      !call TWISSMATCH(1, 2.35, 36.5, 1., 1)
      !call TWISSMATCH(3, 2.35, 36.5, 1., 1)
      ! mcat-dtl-tank1-focus
      ! MCAT-verb: dtl-noquads-focus
      !nlines: 7
      !call dr(70.0,'center-tank2')
      !call twissmatch(1, 0.0, 1.5*bx1, 1., 1)
      !call twissmatch(3, 0.0, 1.5*bx1, 1., 1)
      !call fit(1,1,6,0.0,1.,1)
      !call fit(1,5,5,0.0,1.,1)
      !call fit(1,6,5,0.0,1.,1)
      !  return
      ! MCAT-verb: dtl-tank1-focus
      !nlines: 4
      !call dr(16.373,'half-tank1')
      !call twissmatch(1, 0.0, bx1, 1., 1)
      !call twissmatch(3, 0.0, bx1, 1., 1)
      !  return
      ! endOf_mebt_db0
      ! dtl-fringeQ-outer
      call fringeQ(0.4958,-0.2414,0.2134,-0.5207)
      ! mcat-mass-dtl
      ! MCAT Mass Specification (amu)
      qmass=pmass/931.595
      ! MCAT dtl beta-function
      bdtl=10.
      call marker('DTL:LPM0')
      call marker('DTL:FC0')
      call drift(0.07322,".")
      ! RF cavity/linac: ISAC1:DTL1
      call linacn(101,164,None,32.6,1.0608e+08,None,'ISAC1:DTL1')
      call drift(0.52678,".")
      ! mcat-dtl-tank1-energy
      ! MCAT-verb: set-dtl-tank1-energy
      !nlines: 5
      !call dr(1736.3619,'harp0-drift')
      !call fit(1,6,6,0.,10.,1)
      !energerr0=0.238*qmass - energk
      !call fitarb(0.0,energerr0,10.,1)
      !  return
      ! MCAT-verb: adjust-dtl-tank1-longitudinal
      !nlines: 4
      !call dr(1736.3619,'harp0-drift')
      !call fit(1,6,6,0.,1.,1)
      !call fit(1,6,5,0.,1.,1)
      !  return
      call drift(2.15519,".")

      call genshift(AGX9,0.0,AGY9,0.0)
      ! Magnetic quadrupole DTL:Q1
      call mquad(None,1.1988,5.8,wo,'DTL:Q1')
      call genshift(-AGX9,0.0,-AGY9,0.0)

      call drift(4.84481,".")
      ! dtl-fringeQ-inner
      call fringeQ(0.4504,-0.2316,0.1698,-0.4807)
      call drift(0.90364,".")

      call genshift(AGX10,0.0,AGY10,0.0)
      ! Magnetic quadrupole DTL:Q2
      call mquad(None,1.1988,8.7,wo,'DTL:Q2')
      call genshift(-AGX10,0.0,-AGY10,0.0)

      call drift(5.29636,".")
      ! dtl-fringeQ-outer
      call fringeQ(0.4958,-0.2414,0.2134,-0.5207)
      call drift(0.05364,".")
      ! Steerer DTL:XCB2
      call genshift(0.,XCB3,0.,0.)
      ! Steerer DTL:YCB2
      call genshift(0.,0.,0.,YCB5)
      call drift(0.39845,".")

      call genshift(AGX11,0.0,AGY11,0.0)
      ! Magnetic quadrupole DTL:Q3
      call mquad(None,1.1988,5.8,wo,'DTL:Q3')
      call genshift(-AGX11,0.0,-AGY11,0.0)

      call drift(2.04576,".")
      ! start-buncher1
      call drift(5e-06,".")
      ! RF cavity/linac: ISAC1:BUNCH1
      call linacn(102,1001,None,13.050,1.0608e+08,None,'ISAC1:BUNCH1')
      call drift(0.752135,".")
      ! mcat-dtl-bunch1-energy
      ! MCAT-verb: set-buncher1
      !nlines: 5
      !call dr(1000.0,'.')
      !call fit(1,6,6,0.,1.,1)
      !energerr=0.251*qmass-energk
      !call fitarb(0.0,energerr,10.,1)
      !  return
      call marker('DTL:LPM3')
      call marker('DTL:FC3')
      call drift(3.9,".")
      ! mcat-dtl-triplet1-focus
      ! MCAT-verb: dtl-triplet1-focus
      !nlines: 6
      !qcav=54.775
      !call drift(qcav/2,'.')
      !call twissmatch(1,0.0,bdtl,1.,1)
      !call twissmatch(3,0.0,bdtl,1.,1)
      !call fit(1,1,6,0.,1.,1)
      !  return
      call drift(0.0867,".")
      ! RF cavity/linac: ISAC1:DTL2
      call linacn(103,276,None,55.0,1.0608e+08,None,'ISAC1:DTL2')
      call drift(1.0133,".")
      ! mcat-dtl-tank2-energy
      ! MCAT-verb: set-dtl-tank2-energy
      !nlines: 5
      !call dr(1649.2697,'harp0-drift')
      !call fit(1,6,6,0.,1.,1)
      !energerr=0.461*qmass-energk
      !call fitarb(0.0,energerr,1.,1)
      !  return
      call drift(1.4149,".")

      call genshift(AGX12,0.0,AGY12,0.0)
      ! Magnetic quadrupole DTL:Q4
      call mquad(None,1.1988,5.8,wo,'DTL:Q4')
      call genshift(-AGX12,0.0,-AGY12,0.0)

      call drift(4.7851,".")
      ! dtl-fringeQ-inner
      call fringeQ(0.4504,-0.2316,0.1698,-0.4807)
      call drift(0.9634,".")

      call genshift(AGX13,0.0,AGY13,0.0)
      ! Magnetic quadrupole DTL:Q5
      call mquad(None,1.1988,8.7,wo,'DTL:Q5')
      call genshift(-AGX13,0.0,-AGY13,0.0)

      call drift(5.35,".")
      ! dtl-fringeQ-outer
      call fringeQ(0.4958,-0.2414,0.2134,-0.5207)
      ! Steerer DTL:XCB5
      call genshift(0.,XCB4,0.,0.)
      ! Steerer DTL:YCB5
      call genshift(0.,0.,0.,YCB6)
      call drift(0.3984,".")

      call genshift(AGX14,0.0,AGY14,0.0)
      ! Magnetic quadrupole DTL:Q6
      call mquad(None,1.1988,5.8,wo,'DTL:Q6')
      call genshift(-AGX14,0.0,-AGY14,0.0)

      call drift(2.04794,".")
      ! RF cavity/linac: ISAC1:BUNCH2
      call linacn(104,1001,None,14.850,1.0608e+08,None,'ISAC1:BUNCH2')
      call drift(0.390145,".")
      call marker('DTL:LPM6')
      call marker('DTL:FC6')
      call drift(4.0,".")
      ! mcat-dtl-triplet2-focus
      ! MCAT-verb: dtl-triplet2-focus
      !nlines: 6
      !qcav=81.775
      !call drift(qcav/2,'.')
      !call twissmatch(1,0.0,bdtl,1.,1)
      !call twissmatch(3,0.0,bdtl,1.,1)
      !call fit(1,1,6,0.,1.,1)
      !  return
      call drift(0.3474,".")
      ! RF cavity/linac: ISAC1:DTL3
      call linacn(105,411,None,82.0,1.0608e+08,None,'ISAC1:DTL3')
      call drift(0.1526,".")
      ! mcat-dtl-tank3-energy
      ! MCAT-verb: set-dtl-tank3-energy
      !nlines: 5
      !call dr(1538.4067,'harp0-drift')
      !call fit(1,6,6,0.0,1.,1)
      !energerr=0.803*qmass-energk
      !call fitarb(0.0,energerr,1.,1)
      !  return
      call drift(2.277,".")

      call genshift(AGX15,0.0,AGY15,0.0)
      ! Magnetic quadrupole DTL:Q7
      call mquad(None,1.1988,5.8,wo,'DTL:Q7')
      call genshift(-AGX15,0.0,-AGY15,0.0)

      call drift(1.3984,".")
      ! dtl-fringeQ-inner
      call fringeQ(0.4504,-0.2316,0.1698,-0.4807)
      call drift(4.35,".")

      call genshift(AGX16,0.0,AGY16,0.0)
      ! Magnetic quadrupole DTL:Q8
      call mquad(None,1.1988,8.7,wo,'DTL:Q8')
      call genshift(-AGX16,0.0,-AGY16,0.0)

      call drift(5.35,".")
      ! dtl-fringeQ-outer
      call fringeQ(0.4958,-0.2414,0.2134,-0.5207)
      ! Steerer DTL:XCB8
      call genshift(0.,XCB5,0.,0.)
      ! Steerer DTL:YCB8
      call genshift(0.,0.,0.,YCB7)
      call drift(0.3985,".")

      call genshift(AGX17,0.0,AGY17,0.0)
      ! Magnetic quadrupole DTL:Q9
      call mquad(None,1.1988,5.8,wo,'DTL:Q9')
      call genshift(-AGX17,0.0,-AGY17,0.0)

      call drift(2.04791,".")
      ! RF cavity/linac: ISAC1:BUNCH3
      call linacn(106,1001,None,17.349,1.0608e+08,None,'ISAC1:BUNCH3')
      call drift(1.52821,".")
      call marker('DTL:LPM9')
      call drift(0.08,".")
      call marker('DTL:FC9')
      call drift(3.02,".")
      ! mcat-dtl-triplet3-focus
      ! MCAT-verb: dtl-triplet3-focus
      !nlines: 6
      !qcav=94.775
      !call drift(qcav/2,'.')
      !call twissmatch(1,0.0,bdtl,1.,1)
      !call twissmatch(3,0.0,bdtl,1.,1)
      !call fit(1,1,6,0.,1.,1)
      !  return
      call drift(0.1092,".")
      ! RF cavity/linac: ISAC1:DTL4
      call linacn(107,476,None,95.0,1.0608e+08,None,'ISAC1:DTL4')
      call drift(0.7908,".")
      ! mcat-dtl-tank4-energy
      ! MCAT-verb: set-dtl-tank4-energy
      !nlines: 5
      !call dr(1398.0497,'harp0-drift')
      !call fit(1,6,6,0.0,1.,1)
      !energerr=1.15*qmass-energk
      !call fitarb(0.0,energerr,1.,1)
      !  return
      call drift(1.6386,".")

      call genshift(AGX18,0.0,AGY18,0.0)
      ! Magnetic quadrupole DTL:Q10
      call mquad(None,1.1988,5.8,wo,'DTL:Q10')
      call genshift(-AGX18,0.0,-AGY18,0.0)

      call drift(5.7485,".")
      ! dtl-fringeQ-inner
      call fringeQ(0.4504,-0.2316,0.1698,-0.4807)

      call genshift(AGX19,0.0,AGY19,0.0)
      ! Magnetic quadrupole DTL:Q11
      call mquad(None,1.1988,8.7,wo,'DTL:Q11')
      call genshift(-AGX19,0.0,-AGY19,0.0)

      call drift(5.35,".")
      ! dtl-fringeQ-outer
      call fringeQ(0.4958,-0.2414,0.2134,-0.5207)
      ! Steerer DTL:XCB11
      call genshift(0.,XCB6,0.,0.)
      ! Steerer DTL:YCB11
      call genshift(0.,0.,0.,YCB8)
      call drift(0.3984,".")

      call genshift(AGX20,0.0,AGY20,0.0)
      ! Magnetic quadrupole DTL:Q12
      call mquad(None,1.1988,5.8,wo,'DTL:Q12')
      call genshift(-AGX20,0.0,-AGY20,0.0)

      call drift(3.5645,".")
      call marker('DTL:LPM12')
      call marker('DTL:FC12')
      call drift(5.6511,".")
      ! mcat-dtl-triplet4-focus
      ! MCAT-verb: dtl-triplet4-focus
      !nlines: 6
      !qcav=102.776
      !call drift(qcav,'.')
      !call twissmatch(1,0.0,bdtl,1.,1)
      !call twissmatch(3,0.0,bdtl,1.,1)
      !call fit(1,1,6,0.,1.,1)
      !  return
      ! RF cavity/linac: ISAC1:DTL5
      call linacn(108,516,None,102.77,1.0608e+08,None,'ISAC1:DTL5')
      ! mcat-dtl-tank5-energy
      ! MCAT-verb: set-dtl-tank5-energy
      !nlines: 5
      !call dr(1259.7302,'harp0-drift')
      !call fit(1,6,6,0.,1.,1)
      !energerr=1.53*qmass-energk
      !call fitarb(0.0,energerr,1.,1)
      !  return
      ! endOf_dtl_db0
      ! start_of_hebt_db0
      call drift(6.4599,".")
      ! startOf_t3d_tune_hebt
      call drift(5.7752,".")
      ! HEBT:IV0
      call drift(9.78465,".")
      call marker('HEBT:RPM0')
      call drift(8.26205,".")
      call marker('HEBT:FC0')
      call drift(20.678,".")
      ! Steerer HEBT:XCB0
      call genshift(0.,XCB7,0.,0.)
      ! Steerer HEBT:YCB0
      call genshift(0.,0.,0.,YCB9)
      call drift(9.04023,".")
      ! fringeQ
      call fringeQ(0.48, -0.29, 0.20, -0.48)
      call drift(2.3599,".")

      call genshift(AGX21,0.0,AGY21,0.0)
      ! Magnetic quadrupole HEBT:Q1
      call mquad(None,2.6,18.2,wo,'HEBT:Q1')
      call genshift(-AGX21,0.0,-AGY21,0.0)

      call drift(18.8005,".")

      call genshift(AGX22,0.0,AGY22,0.0)
      ! Magnetic quadrupole HEBT:Q2
      call mquad(None,2.6,18.2,wo,'HEBT:Q2')
      call genshift(-AGX22,0.0,-AGY22,0.0)

      call drift(72.8768,".")
      ! Steerer HEBT:XCB2
      call genshift(0.,XCB8,0.,0.)
      ! Steerer HEBT:YCB2
      call genshift(0.,0.,0.,YCB10)
      call drift(11.4227,".")

      call genshift(AGX23,0.0,AGY23,0.0)
      ! Magnetic quadrupole HEBT:Q3
      call mquad(None,2.6,18.2,wo,'HEBT:Q3')
      call genshift(-AGX23,0.0,-AGY23,0.0)

      call drift(27.9,".")
      ! HEBT:Q4
      call drift(27.9001,".")

      call genshift(AGX24,0.0,AGY24,0.0)
      ! Magnetic quadrupole HEBT:Q5
      call mquad(None,2.6,18.2,wo,'HEBT:Q5')
      call genshift(-AGX24,0.0,-AGY24,0.0)

      call drift(68.9,".")
      ! mcat-set-hebtrpm5-dsb
      ! MCAT-verb: mcat-set-hebtrpm5-dsb
      ! nlines: 3
      ! call twissmatch(1, 0., 20.0, 1., 1)
      ! call twissmatch(3, 0., 20.0, 1., 1)
      ! return
      ! mcat-set-hebtrpm5-focus
      ! MCAT-verb: set-hebt-rpm5-focus
      !nlines: 3
      !call twissmatch(1,0.,bdtl,1.,1)
      !call twissmatch(3,0.,bdtl,1.,1)
      !  return
      call marker('HEBT:RPM5')
      ! Element of unknown type: strp
      call drift(10.54,".")
      call marker('HEBT:FC5')
      call print_transfer_matrix
      return
      end
