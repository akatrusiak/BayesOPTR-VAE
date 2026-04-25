      SUBROUTINE TSYSTEM
      COMMON/SCPARM/QSC,ISC,CMPS
      COMMON/MOM/P,BRHO,pMASS,ENERGK,GSQ,ENERGKi,charge,current
      COMMON/PRINT/IPRINT
      COMMON/SS/SX(13,6)
      COMMON/BLOC1/EAT1,QM1,QM2,QM3,EAB1,EAT2,QM4,QM5,QM6,EAB2,EAT3,QM7,
     &QM8,QM9,EAB3,EAT4,QM10,QM11,QM12,EAT5,QM13,QM14,QM15,QM16,QM17,QM1
     &8,QM19,QM20,QM21,QM22,QM23,QM24
       ! MCAT DTL/HEBT2 tune


      CMPS=2.  ! Number of cm per step, for plotting only
      wo=1.0 ! Weight aberration from optical elements

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
!      call linacn(100,164,0.00646219*RFA1*RFA2 -
!     &0.0909302*RFA2,32.6,1.0608e+08,1.0*RFP1,'ISAC1:DTL1')
      call drift(32.6/2,'.') 
      qdeltaET1=(qmass*EAT1-energki)/charge !effective voltage in MV
      call rfgap(qdeltaET1,90.0,1.0608e+08,0)
      call drift(32.6/2,'.') 
      call drift(0.52678,".")
      call drift(2.15519,".")
      !return
      ! Magnetic quadrupole DTL:Q1
      call mquad(0.9487898*tanh(6.080925e-13*QM1**5.0 +
     &4.098666e-08*QM1**3.0 + 0.00497265*QM1),1.1988,5.8,wo,'DTL:Q1')
      call drift(4.84481,".")
      ! dtl-fringeQ-inner
      call fringeQ(0.4504,-0.2316,0.1698,-0.4807)
      call drift(0.90364,".")
      ! Magnetic quadrupole DTL:Q2
      call mquad(-1.005595*tanh(5.842323e-13*QM2**5.0 +
     &4.001401e-08*QM2**3.0 + 0.004933*QM2),1.1988,8.7,wo,'DTL:Q2')
      call drift(5.29636,".")
      ! dtl-fringeQ-outer
      call fringeQ(0.4958,-0.2414,0.2134,-0.5207)
      call drift(0.05364,".")
      ! DTL:XCB2
      ! DTL:YCB2
      call drift(0.39845,".")
      ! Magnetic quadrupole DTL:Q3
      QM3=QM1
      call mquad(0.9487898*tanh(6.083063e-13*QM3**5.0 +
     &4.099531e-08*QM3**3.0 + 0.004973*QM3),1.1988,5.8,wo,'DTL:Q3')
      call drift(2.04577,".")
      ! RF cavity/linac: ISAC1:BUNCH1
!      call linacn(101,1001,0.007986697*RFA3*RFA4 +
!     &0.3061407*RFA4,13.050,1.0608e+08,1.0*RFP2,'ISAC1:BUNCH1')
      call drift(13.050/2,'.')
      qdeltaEB1=qmass*(EAB1-EAT1)/charge !effective voltage in MV
      call rfgap(qdeltaEB1,90.0,1.0608e+08,0)
      call drift(13.050/2,'.')
      call drift(0.752135,".")
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
!      call linacn(102,276,0.0059044*RFA5*RFA6 +
!     &0.112956*RFA6,55.0,1.0608e+08,1.0*RFP3,'ISAC1:DTL2')
      call drift(55.0/2,'.')
      qdeltaET2=qmass*(EAT2-EAB1)/charge !effective voltage in MV
      call rfgap(qdeltaET2,90.0,1.0608e+08,0)
      call drift(55.0/2,'.')
      call drift(1.0133,".")
      call drift(1.4149,".")
      ! Magnetic quadrupole DTL:Q4
      call mquad(-0.9487898*tanh(6.083063e-13*QM4**5.0 +
     &4.099531e-08*QM4**3.0 + 0.004973*QM4),1.1988,5.8,wo,'DTL:Q4')
      call drift(4.7851,".")
      ! dtl-fringeQ-inner
      call fringeQ(0.4504,-0.2316,0.1698,-0.4807)
      call drift(0.9634,".")
      ! Magnetic quadrupole DTL:Q5
      call mquad(1.005595*tanh(5.842323e-13*QM5**5.0 +
     &4.001401e-08*QM5**3.0 + 0.004933*QM5),1.1988,8.7,wo,'DTL:Q5')
      call drift(5.35,".")
      ! dtl-fringeQ-outer
      call fringeQ(0.4958,-0.2414,0.2134,-0.5207)
      ! DTL:XCB5
      ! DTL:YCB5
      call drift(0.3984,".")
      ! Magnetic quadrupole DTL:Q6
      QM6=QM4
      call mquad(-0.9487898*tanh(6.083063e-13*QM6**5.0 +
     &4.099531e-08*QM6**3.0 + 0.004973*QM6),1.1988,5.8,wo,'DTL:Q6')
      call drift(2.04794,".")
      ! RF cavity/linac: ISAC1:BUNCH2
!      call linacn(103,1001,-1.5917e-06*RFA7**2.0*RFA8 +
!     &0.008996733*RFA7*RFA8 -
!     &0.718549*RFA8,14.850,1.0608e+08,1.0*RFP4,'ISAC1:BUNCH2')
      call drift(14.850/2,'.')
      qdeltaEB2=qmass*(EAB2-EAT2)/charge !effective voltage in MV
      call rfgap(qdeltaEB2,90.0,1.0608e+08,0)
      call drift(14.850/2,'.')
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
!      call linacn(104,411,0.00587736*RFA10*RFA9 -
!     &0.15297*RFA10,82.0,1.0608e+08,1.0*RFP5,'ISAC1:DTL3')
      call drift(82.0/2,'.')
      qdeltaET3=qmass*(EAT3-EAB2)/charge !effective voltage in MV
      call rfgap(qdeltaET3,90.0,1.0608e+08,0)
      call drift(82.0/2,'.')
      call drift(0.1526,".")
      !return
      call drift(2.277,".")
      ! Magnetic quadrupole DTL:Q7
      call mquad(0.9487898*tanh(6.083063e-13*QM7**5.0 +
     &4.099531e-08*QM7**3.0 + 0.004973*QM7),1.1988,5.8,wo,'DTL:Q7')
      call drift(1.3984,".")
      ! dtl-fringeQ-inner
      call fringeQ(0.4504,-0.2316,0.1698,-0.4807)
      call drift(4.35,".")
      ! Magnetic quadrupole DTL:Q8
      call mquad(-1.005595*tanh(5.842323e-13*QM8**5.0 +
     &4.001401e-08*QM8**3.0 + 0.004933*QM8),1.1988,8.7,wo,'DTL:Q8')
      call drift(5.35,".")
      ! dtl-fringeQ-outer
      call fringeQ(0.4958,-0.2414,0.2134,-0.5207)
      ! DTL:XCB8
      ! DTL:YCB8
      call drift(0.3985,".")
      ! Magnetic quadrupole DTL:Q9
      QM9=QM7
      call mquad(0.9487898*tanh(6.083063e-13*QM9**5.0 +
     &4.099531e-08*QM9**3.0 + 0.004973*QM9),1.1988,5.8,wo,'DTL:Q9')
      call drift(2.04791,".")
      ! RF cavity/linac: ISAC1:BUNCH3
!      call linacn(105,1001,0.006496419*RFA11*RFA12 +
!     &0.1508214*RFA12,17.349,1.0608e+08,1.0*RFP6,'ISAC1:BUNCH3')
      call drift(17.349/2,'.')
      qdeltaEB3=qmass*(EAB3-EAT3)/charge !effective voltage in MV
      call rfgap(qdeltaEB3,90.0,1.0608e+08,0)
      call drift(17.349/2,'.')
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
!      call linacn(106,476,0.00459821*RFA13*RFA14 -
!     &0.126286*RFA14,95.0,1.0608e+08,1.0*RFP7,'ISAC1:DTL4')
      call drift(95.0/2,'.')
      qdeltaET4=qmass*(EAT4-EAB3)/charge !effective voltage in MV
      call rfgap(qdeltaET4,90.0,1.0608e+08,0)
      call drift(95.0/2,'.')
      call drift(0.7908,".")
      call drift(1.6386,".")
      ! Magnetic quadrupole DTL:Q10
      call mquad(-0.9487898*tanh(6.083063e-13*QM10**5.0 +
     &4.099531e-08*QM10**3.0 + 0.004973*QM10),1.1988,5.8,wo,'DTL:Q10')
      call drift(5.7485,".")
      ! dtl-fringeQ-inner
      call fringeQ(0.4504,-0.2316,0.1698,-0.4807)
      ! Magnetic quadrupole DTL:Q11
      call mquad(1.005595*tanh(5.842323e-13*QM11**5.0 +
     &4.001401e-08*QM11**3.0 + 0.004933*QM11),1.1988,8.7,wo,'DTL:Q11')
      call drift(5.35,".")
      ! dtl-fringeQ-outer
      call fringeQ(0.4958,-0.2414,0.2134,-0.5207)
      ! DTL:XCB11
      ! DTL:YCB11
      call drift(0.3984,".")
      ! Magnetic quadrupole DTL:Q12
      QM12=QM10
      call mquad(-0.9487898*tanh(6.083063e-13*QM12**5.0 +
     &4.099531e-08*QM12**3.0 + 0.004973*QM12),1.1988,5.8,wo,'DTL:Q12')
      call drift(3.5645,".")
      call marker('DTL:LPM12')
      call marker('DTL:FC12')
      call drift(5.6511,".")
      ! mcat-dtl-triplet4-focus
      ! MCAT-verb: dtl-triplet4-focus
      !nlines: 6
      !qcav=102.776
      !call drift(qcav/2.+40.0,'.')
      !call twissmatch(1,0.0,bdtl,1.,1)
      !call twissmatch(3,0.0,bdtl,1.,1)
      !call fit(1,1,6,0.,1.,1)
      !  return
      ! RF cavity/linac: ISAC1:DTL5
!      call linacn(107,516,0.00422774*RFA15*RFA16 -
!     &0.319679*RFA16,102.77,1.0608e+08,1.0*RFP8,'ISAC1:DTL5')
      call dr(102.77/2,'.')
      qdeltaET5=qmass*(EAT5-EAT4)/charge !effective voltage in MV
      call rfgap(qdeltaET5,90.0,1.0608e+08,0)
      call dr(102.77/2,'.')
      !return
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
      ! HEBT:XCB0
      ! HEBT:YCB0
      call drift(9.04023,".")
      ! fringeQ
      call fringeQ(0.48, -0.29, 0.20, -0.48)
      call drift(2.3599,".")
      ! Magnetic quadrupole HEBT:Q1
      call mquad(3.19953e-10*QM13**5.0 - 5.03293e-08*QM13**4.0 +
     &2.98602e-06*QM13**3.0 - 7.328e-05*QM13**2.0 - 0.0086354*QM13 -
     &0.0043,2.6,18.2,wo,'HEBT:Q1')
      call drift(18.8005,".")
      ! Magnetic quadrupole HEBT:Q2
      call mquad(-3.19953e-10*QM14**5.0 + 5.03293e-08*QM14**4.0 -
     &2.98602e-06*QM14**3.0 + 7.328e-05*QM14**2.0 + 0.0086354*QM14 +
     &0.0043,2.6,18.2,wo,'HEBT:Q2')
      call drift(72.8768,".")
      ! HEBT:XCB2
      ! HEBT:YCB2
      call drift(11.4227,".")
      ! Magnetic quadrupole HEBT:Q3
      call mquad(3.19953e-10*QM15**5.0 - 5.03293e-08*QM15**4.0 +
     &2.98602e-06*QM15**3.0 - 7.328e-05*QM15**2.0 - 0.0086354*QM15 -
     &0.0043,2.6,18.2,wo,'HEBT:Q3')
      call drift(27.9,".")
      ! HEBT:Q4
      call drift(27.9001,".")
      ! Magnetic quadrupole HEBT:Q5
      call mquad(-3.19953e-10*QM16**5.0 + 5.03293e-08*QM16**4.0 -
     &2.98602e-06*QM16**3.0 + 7.328e-05*QM16**2.0 + 0.0086354*QM16 +
     &0.0043,2.6,18.2,wo,'HEBT:Q5')
      call drift(68.9,".")
      ! mcat-set-hebtrpm5-focus
      ! MCAT-verb: set-hebt-rpm5-focus
      !nlines: 3
      !call twissmatch(1,0.,5*bdtl,100.,1)
      !call twissmatch(3,0.,5*bdtl,100.,1)
      !return
      call marker('HEBT:RPM5')
      ! Element of unknown type: strp
      call drift(10.54,".")
      call marker('HEBT:FC5')
      ! Element of unknown type: scd
      call drift(15.1293,".")
      ! Element of unknown type: psid
      call drift(12.9133,".")
      ! HEBT:IV8
      call drift(37.2742,".")
      ! HEBT:XCB5
      ! HEBT:YCB5
      call drift(14.0432,".")
      ! Magnetic quadrupole HEBT:Q6
      call mquad(3.19953e-10*QM17**5.0 - 5.03293e-08*QM17**4.0 +
     &2.98602e-06*QM17**3.0 - 7.328e-05*QM17**2.0 - 0.0086354*QM17 -
     &0.0043,2.6,18.2,wo,'HEBT:Q6')
      call drift(8.94,".")
      ! fringeQ
      call fringeQ(0.52, -0.25, 0.27, -0.48)
      call drift(10.635,".")
      ! mcat-set-hebt-buncher-focus
      ! MCAT-verb: set-hebt-buncher-focus
      !nlines: 1
      !call twissmatch(3,0.0,5.0*bdtl,1000.,1)
      ! Magnetic quadrupole HEBT:Q7
      call mquad(-2.01307e-10*QM18**5.0 + 3.34088e-08*QM18**4.0 -
     &2.1168e-06*QM18**3.0 + 5.5318e-05*QM18**2.0 + 0.0087331*QM18 +
     &0.00442,2.6,33.15,wo,'HEBT:Q7')
      call drift(6.215,".")
      ! fringeQ
      call fringeQ(0.48, -0.29, 0.20, -0.48)
      call drift(13.36,".")
      ! Magnetic quadrupole HEBT:Q8
      call mquad(3.19953e-10*QM19**5.0 - 5.03293e-08*QM19**4.0 +
     &2.98602e-06*QM19**3.0 - 7.328e-05*QM19**2.0 - 0.0086354*QM19 -
     &0.0043,2.6,18.2,wo,'HEBT:Q8')
      call drift(13.36,".")
      ! HEBT:XCB8
      ! HEBT:YCB8
      call drift(10.08,".")
      ! Element of unknown type: col
      call drift(9.8153,".")
      ! endOf_hebt_db0
      call drift(167.143,".")
      ! HEBT1:MB0-drift
      call drift(152.857,".")
      call marker('HEBT:RPM8D')
      ! HEBT:IV8D
      call drift(80.0,".")
      ! fringeQ
      call fringeQ(0.52, -0.25, 0.27, -0.48)
      call drift(18.3174,".")
      ! Magnetic quadrupole HEBT:Q9
      QM20=QM18
      call mquad(2.01307e-10*QM20**5.0 - 3.34088e-08*QM20**4.0 +
     &2.1168e-06*QM20**3.0 - 5.5318e-05*QM20**2.0 - 0.0087331*QM20 -
     &0.00442,2.6,33.15,wo,'HEBT:Q9')
      ! mcat-set-hebt-buncher-focus
      ! MCAT-verb: set-hebt-buncher-focus
      !nlines: 1
      !call twissmatch(1,0.0,5.0*bdtl,1000.,1)
      call drift(8.5326,".")
      ! fringeQ
      call fringeQ(0.48, -0.29, 0.20, -0.48)
      call drift(11.0424,".")
      ! Magnetic quadrupole HEBT:Q10
      QM21=QM17
      call mquad(-3.19953e-10*QM21**5.0 + 5.03293e-08*QM21**4.0 -
     &2.98602e-06*QM21**3.0 + 7.328e-05*QM21**2.0 + 0.0086354*QM21 +
     &0.0043,2.6,18.2,wo,'HEBT:Q10')
      call drift(9.8447,".")
      ! RF cavity/linac: ISAC1:HEBT11
      call drift(70.383,'.') !disabled for now, 2023-08-09
      call drift(9.6719,".")
      call marker('HEBT:LPM10')
      ! mcat-set-hebt-buncher-focus
      ! MCAT-verb: set-hebt-buncher-focus
      !nlines: 3
      !call twissmatch(1,0.0,5.0*bdtl,500.,1)
      !call twissmatch(3,0.0,5.0*bdtl,500.,1)
      !  return
      ! mcat-adjust-hebt-buncher-focus
      ! MCAT-verb: adjust-hebt-buncher-focus
      !nlines: 3
      !call twissmatch(1,0.0,5.0*bdtl,1000.,1)
      !call twissmatch(3,0.0,5.0*bdtl,1000.,1)
      !  return
      call marker('HEBT:FC10')
      call drift(2.70955,".")
      ! RF cavity/linac: ISAC1:HEBT35
      call drift(39.042,'.') !disabled for now 2023-08-09
      call drift(15.2084,".")
      ! HEBT:XCB10
      ! HEBT:YCB10
      call drift(13.9398,".")
      ! Magnetic quadrupole HEBT:Q11
      call mquad(3.19953e-10*QM22**5.0 - 5.03293e-08*QM22**4.0 +
     &2.98602e-06*QM22**3.0 - 7.328e-05*QM22**2.0 - 0.0086354*QM22 -
     &0.0043,2.6,18.2,wo,'HEBT:Q11')
      call drift(34.8,".")
      ! Magnetic quadrupole HEBT:Q12
      call mquad(-3.19953e-10*QM23**5.0 + 5.03293e-08*QM23**4.0 -
     &2.98602e-06*QM23**3.0 + 7.328e-05*QM23**2.0 + 0.0086354*QM23 +
     &0.0043,2.6,18.2,wo,'HEBT:Q12')
      call drift(48.5545,".")
      ! HEBT:XCB12
      ! HEBT:YCB12
      call drift(26.0658,".")
      ! HEBT:IV12
      call drift(9.6999,".")
      call marker('HEBT:RPM12')
      call drift(8.255,".")
      call marker('HEBT:FC12')
      call drift(36.3248,".")
      ! endOf_hebt_db9
      call drift(0.07366,".")
      ! Magnetic dipole HEBT2:MB0
      call edge(11.25,100.003,22.5,0.0,0.55,0.0,5.53,0.0,wo)
      call bend(100.003,22.5,0.0,'HEBT2:MB0')
      call edge(11.25,100.003,22.5,0.0,0.55,0.0,5.53,0.0,wo)
      call drift(55.0194,".")
      ! Magnetic quadrupole HEBT2:Q1
      call mquad(-0.00029746*QM24,1.0,8.0,wo,'HEBT2:Q1')
      call drift(10.6889,".")
      ! HEBT2:IV1
      call drift(9.1338,".")
      ! mcat-set-hebt2-rpm1-1
      ! MCAT-verb: set-hebt2-rpm1-focus
      !nlines: 2
      !call twissmatch(1,0.,4.0*bdtl,4500.,1)
      !call twissmatch(3,0.,4.0*bdtl,4500.,1)
      call marker('HEBT2:RPM1')
      call drift(5.1791,".")
      call marker('HEBT2:XSLIT1')
      call marker('HEBT2:FC1')
      call print_transfer_matrix
      return
      end

