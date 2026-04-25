      SUBROUTINE TSYSTEM
      COMMON/SCPARM/QSC,ISC,CMPS
      COMMON/MOM/P,BRHO,pMASS,ENERGK,GSQ,ENERGKi,charge,current
      COMMON/PRINT/IPRINT
      COMMON/SS/SX(13,6)
      COMMON/BLOC1/QM1,QM2,QF1,QFM1,QF2,QFM2,QM3,QM4,QM5,QM6,QM7,QM8,EAT
     &1,QM9,QM10,QM11,EAB1,EAT2,QM12,QM13,QM14,EAB2,EAT3,QM15,QM16,QM17,
     &EAB3,EAT4,QM18,QM19,QM20,EAT5,QM21,QM22,QM23,QM24
      COMMON/BLOC1/QM25,QM26,QM27,QM28,QM29,QM30,QM31,QHMB0F,QHMB0M,QM32
     &,QHMB1F,QHMB1M,QM33,QM34,QM35,QM36,QM37,QM38


      CMPS=1.  ! Number of cm per step, for plotting only
      wo=10.0 ! Weight aberration from optical elements

      !DIPOLE PARAMETERS FOR MCAT 2024-05-31
      !HEBT2:MB0:FLD:SETP
      !HEBT2:MB0:FLD:MODE
      !HEBT2:MB1:FLD:SETP
      !HEBT2:MB1:FLD:MODE






      call twissfind(1,ax1,bx1)
      call twissfind(3,ay1,by1)

      call marker('MEBT:RPM5')
      ! mcat-foil-match-twissfind
      call drift(2.0041,".")
      ! mcat-mebt-set-chopslit
      call fringeQ(0.368,-0.222,0.116,-0.422)
      bfoc=bx1
      ! mebt-frame-rotation
      call RT(45.0)
      ! MCAT Mass Specification (amu)
      qmass=pmass/931.595
      ! MEBT:CHOPSLIT
      call drift(16.1519,".")
      ! Element of unknown type: strp
      call drift(14.3078,".")
      ! MEBT:IV6
      call drift(11.855,".")
      ! Magnetic quadrupole MEBT:Q6
      call mquad(-0.798395*tanh(4.144494e-11*QM1**5.0 +
     &5.161117e-07*QM1**3.0 + 0.0115688*QM1),2.547,17.89,wo,'MEBT:Q6')
      call drift(12.1176,".")
      ! Magnetic quadrupole MEBT:Q7
      call mquad(0.798395*tanh(4.144494e-11*QM2**5.0 +
     &5.161117e-07*QM2**3.0 + 0.0115688*QM2),2.547,17.89,wo,'MEBT:Q7')
      call drift(11.9884,".")
      ! MEBT:YCB7A
      call drift(24.0675,".")
      ! Magnetic dipole MEBT:MB1
      call edge(22.5,29.9975,45.0,0.0,0.55,0.0,5.72,0.0,wo)
      call bend(29.9975,45.0,0.0,'MEBT:MB1')
      call edge(22.5,29.9975,45.0,0.0,0.55,0.0,5.72,0.0,wo)
      call drift(24.8355,".")
      call marker('MEBT:RPM7')
      call drift(5.1677,".")
      ! MCAT-verb: mebt-corner-single-achromat
      !nlines: 1
      !call fit(1,1,1,0.0,50000.,1)
      ! MEBT:XSLIT7
      call marker('MEBT:FC7')
      call drift(29.9988,".")
      ! Magnetic dipole MEBT:MB2
      call edge(22.5,29.9975,45.0,0.0,0.55,0.0,5.72,0.0,wo)
      call bend(29.9975,45.0,0.0,'MEBT:MB2')
      call edge(22.5,29.9975,45.0,0.0,0.55,0.0,5.72,0.0,wo)
      call drift(24.5577,".")
      ! MEBT:YCB7B
      call drift(11.4987,".")
      ! Magnetic quadrupole MEBT:Q8
      QM3=QM2
      call mquad(0.798395*tanh(4.144494e-11*QM3**5.0 +
     &5.161117e-07*QM3**3.0 + 0.0115688*QM3),2.547,17.89,wo,'MEBT:Q8')
      call drift(12.115,".")
      ! Magnetic quadrupole MEBT:Q9
      QM4=QM1
      call mquad(-0.798395*tanh(4.144494e-11*QM4**5.0 +
     &5.161117e-07*QM4**3.0 + 0.0115688*QM4),2.547,17.89,wo,'MEBT:Q9')
      call drift(12.9874,".")
      ! Element of unknown type: sid
      call marker('MEBT:FC9')
      call drift(2.8829,".")
      ! MCAT-verb: mebt-corner-single-achromat
      !nlines: 4
      !call dr(9.96,'half-rebuncher')
      !call twissmatch(1, 0.0, bfoc, 1000., 1)
      !call twissmatch(3, 0.0, bfoc, 1000., 1)
      !  return
      call drift(5e-05,".")
      ! RF cavity/linac: ISAC1:MEBT
      call drift(19.926,'.')
      call drift(3.09995,".")
      call drift(3.8485,".")
      ! MEBT:XCB9
      ! MEBT:YCB9
      call drift(10.9363,".")
      ! Magnetic quadrupole MEBT:Q10
      call mquad(0.798395*tanh(4.144494e-11*QM5**5.0 +
     &5.161117e-07*QM5**3.0 + 0.0115688*QM5),2.547,17.89,wo,'MEBT:Q10')
      call drift(11.8788,".")
      ! Magnetic quadrupole MEBT:Q11
      call mquad(-0.798395*tanh(4.144494e-11*QM6**5.0 +
     &5.161117e-07*QM6**3.0 + 0.0115688*QM6),2.547,17.89,wo,'MEBT:Q11')
      call drift(5.9394,".")
      ! MEBT:YCB11
      call drift(5.9394,".")
      ! Magnetic quadrupole MEBT:Q12
      call mquad(0.798395*tanh(4.144494e-11*QM7**5.0 +
     &5.161117e-07*QM7**3.0 + 0.0115688*QM7),2.547,17.89,wo,'MEBT:Q12')
      call drift(5.9394,".")
      ! MEBT:XCB12
      call drift(5.9394,".")
      ! Magnetic quadrupole MEBT:Q13
      call mquad(-0.798395*tanh(4.144494e-11*QM8**5.0 +
     &5.161117e-07*QM8**3.0 + 0.0115688*QM8),2.547,17.89,wo,'MEBT:Q13')
      call drift(18.8543,".")
      ! MEBT:IV13
      bdtl=20.
      ! MCAT-verb: dtl-tank1-focus
      ! nlines: 6
      ! qbinj=36.5
      ! qainj=2.35
      ! call drift(0.0,'finesse-location')
      ! call twissmatch(1,qainj,qbinj,10.,1) 
      ! call twissmatch(3,qainj,qbinj,10.,1) 
      ! return
      call drift(4.572,".")
      ! endOf_mebt_db0
      ! dtl-fringeQ-outer
      call fringeQ(0.4958,-0.2414,0.2134,-0.5207)
      ! mcat-mass-dtl
      ! MCAT Mass Specification (amu)
      qmass=pmass/931.595
      ! MCAT dtl beta-function
      call marker('DTL:FC0')
      call drift(0.07322,".")
      ! RF cavity/linac: ISAC1:DTL1
      call drift(32.6/2,'.') 
      qdeltaET1=(qmass*EAT1-energki)/charge !effective voltage in MV
      call rfgap(qdeltaET1,90.0,1.0608e+08,0)
      call drift(32.6/2,'.') 
      call drift(2.68197,".")
      ! Magnetic quadrupole DTL:Q1
      call mquad(0.9487898*tanh(6.080925e-13*QM9**5.0 +
     &4.098666e-08*QM9**3.0 + 0.00497265*QM9),1.1988,5.8,wo,'DTL:Q1')
      call drift(4.84481,".")
      ! dtl-fringeQ-inner
      call fringeQ(0.4504,-0.2316,0.1698,-0.4807)
      call drift(0.90364,".")
      ! Magnetic quadrupole DTL:Q2
      call mquad(-1.005595*tanh(5.842323e-13*QM10**5.0 +
     &4.001401e-08*QM10**3.0 + 0.004933*QM10),1.1988,8.7,wo,'DTL:Q2')
      call drift(5.29636,".")
      ! dtl-fringeQ-outer
      call fringeQ(0.4958,-0.2414,0.2134,-0.5207)
      call drift(0.05364,".")
      ! DTL:XCB2
      ! DTL:YCB2
      call drift(0.39845,".")
      ! Magnetic quadrupole DTL:Q3
      call mquad(0.9487898*tanh(6.083063e-13*QM11**5.0 +
     &4.099531e-08*QM11**3.0 + 0.004973*QM11),1.1988,5.8,wo,'DTL:Q3')
      call drift(2.04576,".")
      ! start-buncher1
      call drift(5e-06,".")
      ! RF cavity/linac: ISAC1:BUNCH1
      call drift(13.050/2,'.')
      qdeltaEB1=qmass*(EAB1-EAT1)/charge !effective voltage in MV
      call rfgap(qdeltaEB1,90.0,1.0608e+08,0)
      call drift(13.050/2,'.')
      call drift(0.752135,".")
      call marker('DTL:FC3')
      call drift(3.9,".")
      ! MCAT-verb: dtl-triplet1-focus
      !nlines: 5
      !qcav=54.775
      !call drift(qcav/2,'.')
      !call twissmatch(1,0.0,bdtl,1000.,1)
      !call twissmatch(3,0.0,bdtl,1000.,1)
      !  return
      call drift(0.0867,".")
      ! RF cavity/linac: ISAC1:DTL2
      call drift(55.0/2,'.')
      qdeltaET2=qmass*(EAT2-EAB1)/charge !effective voltage in MV
      call rfgap(qdeltaET2,90.0,1.0608e+08,0)
      call drift(55.0/2,'.')
      call drift(2.4282,".")
      ! Magnetic quadrupole DTL:Q4
      call mquad(-0.9487898*tanh(6.083063e-13*QM12**5.0 +
     &4.099531e-08*QM12**3.0 + 0.004973*QM12),1.1988,5.8,wo,'DTL:Q4')
      call drift(4.7851,".")
      ! dtl-fringeQ-inner
      call fringeQ(0.4504,-0.2316,0.1698,-0.4807)
      call drift(0.9634,".")
      ! Magnetic quadrupole DTL:Q5
      call mquad(1.005595*tanh(5.842323e-13*QM13**5.0 +
     &4.001401e-08*QM13**3.0 + 0.004933*QM13),1.1988,8.7,wo,'DTL:Q5')
      call drift(5.35,".")
      ! dtl-fringeQ-outer
      call fringeQ(0.4958,-0.2414,0.2134,-0.5207)
      ! DTL:XCB5
      ! DTL:YCB5
      call drift(0.3984,".")
      ! Magnetic quadrupole DTL:Q6
      call mquad(-0.9487898*tanh(6.083063e-13*QM14**5.0 +
     &4.099531e-08*QM14**3.0 + 0.004973*QM14),1.1988,5.8,wo,'DTL:Q6')
      call drift(2.04794,".")
      ! RF cavity/linac: ISAC1:BUNCH2
      call drift(14.850/2,'.')
      qdeltaEB2=qmass*(EAB2-EAT2)/charge !effective voltage in MV
      call rfgap(qdeltaEB2,90.0,1.0608e+08,0)
      call drift(14.850/2,'.')
      call drift(0.390145,".")
      call marker('DTL:FC6')
      call drift(4.0,".")
      ! MCAT-verb: dtl-triplet2-focus
      !nlines: 5
      !qcav=81.775
      !call drift(qcav/2.,'.')
      !call twissmatch(1,0.0,bdtl,1000.,1)
      !call twissmatch(3,0.0,bdtl,1000.,1)
      !  return
      call drift(0.3474,".")
      ! RF cavity/linac: ISAC1:DTL3
      call drift(82.0/2,'.')
      qdeltaET3=qmass*(EAT3-EAB2)/charge !effective voltage in MV
      call rfgap(qdeltaET3,90.0,1.0608e+08,0)
      call drift(82.0/2,'.')
      call drift(2.4296,".")
      ! Magnetic quadrupole DTL:Q7
      call mquad(0.9487898*tanh(6.083063e-13*QM15**5.0 +
     &4.099531e-08*QM15**3.0 + 0.004973*QM15),1.1988,5.8,wo,'DTL:Q7')
      call drift(1.3984,".")
      ! dtl-fringeQ-inner
      call fringeQ(0.4504,-0.2316,0.1698,-0.4807)
      call drift(4.35,".")
      ! Magnetic quadrupole DTL:Q8
      call mquad(-1.005595*tanh(5.842323e-13*QM16**5.0 +
     &4.001401e-08*QM16**3.0 + 0.004933*QM16),1.1988,8.7,wo,'DTL:Q8')
      call drift(5.35,".")
      ! dtl-fringeQ-outer
      call fringeQ(0.4958,-0.2414,0.2134,-0.5207)
      ! DTL:XCB8
      ! DTL:YCB8
      call drift(0.3985,".")
      ! Magnetic quadrupole DTL:Q9
      call mquad(0.9487898*tanh(6.083063e-13*QM17**5.0 +
     &4.099531e-08*QM17**3.0 + 0.004973*QM17),1.1988,5.8,wo,'DTL:Q9')
      call drift(2.04791,".")
      ! RF cavity/linac: ISAC1:BUNCH3
      call drift(17.349/2,'.')
      qdeltaEB3=qmass*(EAB3-EAT3)/charge !effective voltage in MV
      call rfgap(qdeltaEB3,90.0,1.0608e+08,0)
      call drift(17.349/2,'.')
      call drift(1.52821,".")
      call drift(0.08,".")
      call marker('DTL:FC9')
      call drift(3.02,".")
      ! MCAT-verb: dtl-triplet3-focus
      !nlines: 5
      !qcav=94.775
      !call drift(qcav/2,'.')
      !call twissmatch(1,0.0,bdtl,1000.,1)
      !call twissmatch(3,0.0,bdtl,1000.,1)
      !  return
      call drift(0.1092,".")
      ! RF cavity/linac: ISAC1:DTL4
      call drift(95.0/2,'.')
      qdeltaET4=qmass*(EAT4-EAB3)/charge !effective voltage in MV
      call rfgap(qdeltaET4,90.0,1.0608e+08,0)
      call drift(95.0/2,'.')
      call drift(2.4294,".")
      ! Magnetic quadrupole DTL:Q10
      call mquad(-0.9487898*tanh(6.083063e-13*QM18**5.0 +
     &4.099531e-08*QM18**3.0 + 0.004973*QM18),1.1988,5.8,wo,'DTL:Q10')
      call drift(5.7485,".")
      ! dtl-fringeQ-inner
      call fringeQ(0.4504,-0.2316,0.1698,-0.4807)
      ! Magnetic quadrupole DTL:Q11
      call mquad(1.005595*tanh(5.842323e-13*QM19**5.0 +
     &4.001401e-08*QM19**3.0 + 0.004933*QM19),1.1988,8.7,wo,'DTL:Q11')
      call drift(5.35,".")
      ! dtl-fringeQ-outer
      call fringeQ(0.4958,-0.2414,0.2134,-0.5207)
      ! DTL:XCB11
      ! DTL:YCB11
      call drift(0.3984,".")
      ! Magnetic quadrupole DTL:Q12
      call mquad(-0.9487898*tanh(6.083063e-13*QM20**5.0 +
     &4.099531e-08*QM20**3.0 + 0.004973*QM20),1.1988,5.8,wo,'DTL:Q12')
      call drift(3.5645,".")
      call marker('DTL:FC12')
      call drift(5.6511,".")
      ! MCAT-verb: dtl-triplet4-focus
      !nlines: 5
      !qcav=102.776
      !call drift(qcav,'.')
      !call twissmatch(1,0.0,bdtl,1000.,1)
      !call twissmatch(3,0.0,bdtl,1000.,1)
      !  return
      ! RF cavity/linac: ISAC1:DTL5
      call dr(102.77/2,'.')
      qdeltaET5=qmass*(EAT5-EAT4)/charge !effective voltage in MV
      call rfgap(qdeltaET5,90.0,1.0608e+08,0)
      call dr(102.77/2,'.')
      ! endOf_dtl_db0
      ! start_of_hebt_db0
      call drift(6.4599,".")
      ! startOf_t3d_tune_hebt
      call drift(5.7752,".")
      ! HEBT:IV0
      call drift(9.78465,".")
      call marker('HEBT:xRPM0')
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
      call mquad(3.19953e-10*QM21**5.0 - 5.03293e-08*QM21**4.0 +
     &2.98602e-06*QM21**3.0 - 7.328e-05*QM21**2.0 - 0.0086354*QM21 -
     &0.0043,2.6,18.2,wo,'HEBT:Q1')
      call drift(18.8004,".")
      ! MCAT-verb: set-hebt-rpm5-focus
      ! nlines: 3
      !call drift(9.1,'mid-Q2')
      !call fit(1,1,1,0.0,1.,1)
      !call drift(-9.1,'back-out')
      !return
      call drift(5e-05,".")
      ! Magnetic quadrupole HEBT:Q2
      call mquad(-3.19953e-10*QM22**5.0 + 5.03293e-08*QM22**4.0 -
     &2.98602e-06*QM22**3.0 + 7.328e-05*QM22**2.0 + 0.0086354*QM22 +
     &0.0043,2.6,18.2,wo,'HEBT:Q2')
      call drift(72.8768,".")
      ! HEBT:XCB2
      ! HEBT:YCB2
      call drift(11.4227,".")
      ! MCAT-verb: set-hebt-rpm5-focus
      ! nlines: 3
      !call drift(9.1,'mid-Q3')
      !call fit(1,3,3,0.0,1.,1)
      !call drift(-9.1,'back-out')
      !return
      ! Magnetic quadrupole HEBT:Q3
      call mquad(3.19953e-10*QM23**5.0 - 5.03293e-08*QM23**4.0 +
     &2.98602e-06*QM23**3.0 - 7.328e-05*QM23**2.0 - 0.0086354*QM23 -
     &0.0043,2.6,18.2,wo,'HEBT:Q3')
      call drift(27.9,".")
      ! HEBT:Q4
      call drift(27.9001,".")
      ! Magnetic quadrupole HEBT:Q5
      call mquad(-3.19953e-10*QM24**5.0 + 5.03293e-08*QM24**4.0 -
     &2.98602e-06*QM24**3.0 + 7.328e-05*QM24**2.0 + 0.0086354*QM24 +
     &0.0043,2.6,18.2,wo,'HEBT:Q5')
      call drift(68.9,".")
      ! MCAT-verb: set-hebt-rpm5-focus
      ! nlines: 3
      !call twissmatch(1,0.,1.0*bdtl,5000.,1)
      !call twissmatch(3,0.,1.0*bdtl,5000.,1)
      !  return
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
      qs=1.0
      call mquad(qs*(3.19953e-10*QM25**5.0 - 5.03293e-08*QM25**4.0 +
     &2.98602e-06*QM25**3.0 - 7.328e-05*QM25**2.0 - 0.0086354*QM25 -
     &0.0043),2.6,18.2,wo,'HEBT:Q6')
      call drift(8.94,".")
      ! fringeQ
      call fringeQ(0.52, -0.25, 0.27, -0.48)
      call drift(10.635,".")
      ! mcat-set-hebt-buncher-focus
      ! MCAT-verb: set-hebt-buncher-focus
      !nlines: 1
      !call twissmatch(3,0.0,5.0*bdtl,1000.,1)
      ! Magnetic quadrupole HEBT:Q7
      call mquad(qs*(-2.01307e-10*QM26**5.0 + 3.34088e-08*QM26**4.0 -
     &2.1168e-06*QM26**3.0 + 5.5318e-05*QM26**2.0 + 0.0087331*QM26 +
     &0.00442),2.6,33.15,wo,'HEBT:Q7')
      call drift(6.215,".")
      ! fringeQ
      call fringeQ(0.48, -0.29, 0.20, -0.48)
      call drift(13.36,".")
      ! Magnetic quadrupole HEBT:Q8
      call mquad(qs*(3.19953e-10*QM27**5.0 - 5.03293e-08*QM27**4.0 +
     &2.98602e-06*QM27**3.0 - 7.328e-05*QM27**2.0 - 0.0086354*QM27 -
     &0.0043),2.6,18.2,wo,'HEBT:Q8')
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
      ! MCAT-verb: set-rpm8d
      !nlines: 5
      !call drift(-10.0,'.')
      !q8d=5.0*bdtl
      !call twissmatch(1,0.0,q8d,1000.,1)
      !call twissmatch(3,0.0,q8d,1000.,1)
      !return
      ! HEBT:IV8D
      call drift(80.0,".")
      ! fringeQ
      call fringeQ(0.52, -0.25, 0.27, -0.48)
      call drift(18.3174,".")
      !SECOND CONSTRAINT
      ! MCAT-verb: set-buzz1
      !nlines: 2
      !call find(1,1,1,qx9)
      !call find(1,3,3,qy9)
      ! Magnetic quadrupole HEBT:Q9
      !QM28=QM26!UNTIED 2023-10-31
      call mquad(qs*(2.01307e-10*QM28**5.0 - 3.34088e-08*QM28**4.0 +
     &2.1168e-06*QM28**3.0 - 5.5318e-05*QM28**2.0 - 0.0087331*QM28 -
     &0.00442),2.6,33.15,wo,'HEBT:Q9')
      ! mcat-set-hebt-buncher-focus
      ! MCAT-verb: set-hebt-buncher-focus
      !nlines: 1
      !call twissmatch(1,0.0,5.0*bdtl,1000.,1)
      call drift(8.5326,".")
      ! fringeQ
      call fringeQ(0.48, -0.29, 0.20, -0.48)
      call drift(11.0424,".")
      ! MCAT-verb: set-buzz2
      !nlines: 2
      !call find(1,1,1,qx10)
      !call find(1,3,3,qy10)
      ! Magnetic quadrupole HEBT:Q10
      !QM29=QM25!UNTIED 2023-10-31
      call mquad(qs*(-3.19953e-10*QM29**5.0 + 5.03293e-08*QM29**4.0 -
     &2.98602e-06*QM29**3.0 + 7.328e-05*QM29**2.0 + 0.0086354*QM29 +
     &0.0043),2.6,18.2,wo,'HEBT:Q10')
      call drift(9.8447,".")
      ! RF cavity/linac: ISAC1:HEBT11
      call drift(70.383,'.') !disabled for now, 2023-08-09
      call drift(9.6719,".")
      call marker('HEBT:LPM10')
      ! mcat-set-hebt-buncher-focus
      ! MCAT-verb: set-hebt-buncher-focus
      !nlines: 3
      !call twissmatch(1,0.0,5.0*bdtl,1500.,1)
      !call twissmatch(3,0.0,5.0*bdtl,1500.,1)
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
      ! MCAT-verb: set-buzz1
      !nlines: 3
      !call fit(1,1,1,qx9,1000.,1)
      !call fit(1,3,3,qy9,1000.,1)
      !return
      ! Magnetic quadrupole HEBT:Q11
      call mquad(qs*(3.19953e-10*QM30**5.0 - 5.03293e-08*QM30**4.0 +
     &2.98602e-06*QM30**3.0 - 7.328e-05*QM30**2.0 - 0.0086354*QM30 -
     &0.0043),2.6,18.2,wo,'HEBT:Q11')
      call drift(34.8,".")
      ! MCAT-verb: set-buzz2
      !nlines: 3
      !call fit(1,1,1,qx10,1000.,1) 
      !call fit(1,3,3,qy10,1000.,1)
      !return
      ! MCAT-verb: set-buzz2
      !nlines: 2
      !call find(1,1,1,qx12)
      !call find(1,3,3,qy12)
      ! Magnetic quadrupole HEBT:Q12
      call mquad(qs*(-3.19953e-10*QM31**5.0 + 5.03293e-08*QM31**4.0 -
     &2.98602e-06*QM31**3.0 + 7.328e-05*QM31**2.0 + 0.0086354*QM31 +
     &0.0043),2.6,18.2,wo,'HEBT:Q12')
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
      call edge(11.25,-100.003,-22.5,0.0,0.55,0.0,5.53,0.0,wo)
      call bend(-100.003,-22.5,0.0,'HEBT2:MB0')
      call edge(11.25,-100.003,-22.5,0.0,0.55,0.0,5.53,0.0,wo)
      call drift(55.0194,".")
      ! Magnetic quadrupole HEBT2:Q1!ORIG-
      call mquad(qs*(0.00029746*QM32),1.0,8.0,wo,'HEBT2:Q1')
      call drift(10.6889,".")
      ! HEBT2:IV1
      call drift(9.1338,".")
      ! mcat-set-hebt2-rpm1-1
      ! MCAT-verb: set-hebt2-rpm1-focus
      !nlines: 2
      !call twissmatch(1,0.,7.0*bdtl,4500.,1)
      !call twissmatch(3,0.,7.0*bdtl,4500.,1)
      call marker('HEBT2:RPM1')
      call drift(5.1791,".")
      call marker('HEBT2:XSLIT1')
      ! MCAT-verb: set-buzz3
      !nlines: 2
      !call twissmatch(1,0.0,bdtl,5000.,1) 
      !call fit(1,4,3,0.0,1000.,1) 
      call marker('HEBT2:FC1')
      call print_transfer_matrix
      return
      call drift(24.9991,".")
      ! Magnetic quadrupole HEBT2:Q2!ORIG-
      call mquad(qs*(0.00029746*QM32),1.0,8.0,wo,'HEBT2:Q2')
      call drift(54.956,".")
      ! Magnetic dipole HEBT2:MB1
      call edge(11.25,-100.003,-22.5,0.0,0.55,0.0,5.53,0.0,wo)
      call bend(-100.003,-22.5,0.0,'HEBT2:MB1')
      call edge(11.25,-100.003,-22.5,0.0,0.55,0.0,5.53,0.0,wo)
      call drift(60.4108,".")
      ! HEBT2:XCB2
      ! HEBT2:YCB2
      call drift(62.4874,".")
      ! MCAT-verb: set-buzz3
      !nlines: 2
      !call fit(1,1,1,0.6*qx12,1000.,1)
      !call fit(1,3,3,0.6*qy12,1000.,1)
      ! Magnetic quadrupole HEBT2:Q3
      call mquad(qs*(-3.19953e-10*QM33**5.0 + 5.03293e-08*QM33**4.0 -
     &2.98602e-06*QM33**3.0 + 7.328e-05*QM33**2.0 + 0.0086354*QM33 +
     &0.0043),2.6,18.0,wo,'HEBT2:Q3')
      call drift(34.9997,".")
      ! Magnetic quadrupole HEBT2:Q4
      call mquad(qs*(3.19953e-10*QM34**5.0 - 5.03293e-08*QM34**4.0 +
     &2.98602e-06*QM34**3.0 - 7.328e-05*QM34**2.0 - 0.0086354*QM34 -
     &0.0043),2.6,18.0,wo,'HEBT2:Q4')
      call drift(75.9752,".")
      ! mcat-set-hebt2-rpm1-2
      ! MCAT-verb: set-hebt2-rpm1-focus
      !nlines: 3
      !call twissmatch(1,0.,4.0*bdtl,2500.,1)
      !call twissmatch(3,0.,4.0*bdtl,2500.,1)
      !return
      ! MCAT-verb: set-buzz3
      !nlines: 3
      !call fit(1,2,1,0.0,1000.,1) 
      !call fit(1,4,3,0.0,1000.,1)
      !return

      call marker('HEBT2:RPM4')
      call drift(2.9751,".")
      ! Element of unknown type: ffc
      call marker('HEBT2:FC4')
      call drift(16.7331,".")
      ! HEBT2:XCB4
      ! HEBT2:YCB4
      call drift(15.3152,".")
      ! mcat-set-dragon-gas-1 (1 AND 2 ADDED MANUALLY OS 2022-10-27)
      ! MCAT-verb: set-dragon-gas
      !nlines: 2
      !call twissfind(1,axg,bxg)
      !call twissfind(3,ayg,byg)
      ! Magnetic quadrupole HEBT2:Q5
      call mquad(qs*(3.19953e-10*QM35**5.0 - 5.03293e-08*QM35**4.0 +
     &2.98602e-06*QM35**3.0 - 7.328e-05*QM35**2.0 - 0.0086354*QM35 -
     &0.0043),2.6,18.0,wo,'HEBT2:Q5')
      call drift(30.0004,".")
      ! Magnetic quadrupole HEBT2:Q6
      call mquad(qs*(-3.19953e-10*QM36**5.0 + 5.03293e-08*QM36**4.0 -
     &2.98602e-06*QM36**3.0 + 7.328e-05*QM36**2.0 + 0.0086354*QM36 +
     &0.0043),2.6,18.0,wo,'HEBT2:Q6')
      call drift(14.9495,".")
      ! HEBT2:XCB6
      ! HEBT2:YCB6
      call drift(15.05,".")
      ! Magnetic quadrupole HEBT2:Q7
      call mquad(qs*(3.19953e-10*QM37**5.0 - 5.03293e-08*QM37**4.0 +
     &2.98602e-06*QM37**3.0 - 7.328e-05*QM37**2.0 - 0.0086354*QM37 -
     &0.0043),2.6,18.0,wo,'HEBT2:Q7')
      ! mcat-set-dragon-gas-2
      ! MCAT-verb: set-dragon-gas
      !nlines: 2
      !call twissmatch(1,-axg,bxg,25.,1)
      !call twissmatch(3,-ayg,byg,25.,1)
      call drift(29.9995,".")
      ! Magnetic quadrupole HEBT2:Q8
      call mquad(qs*(3.19953e-10*QM38**5.0 - 5.03293e-08*QM38**4.0 +
     &2.98602e-06*QM38**3.0 - 7.328e-05*QM38**2.0 - 0.0086354*QM38 -
     &0.0043),2.6,18.0,wo,'HEBT2:Q8')
      call drift(11.9384,".")
      ! HEBT2:IV8
      call drift(109.062,".")
      ! mcat-set-dragon-gas-3
      ! MCAT-verb: set-dragon-gas
      !nlines: 3
      !call twissmatch(1,0.,1.0*bdtl,10000.,1)
      !call twissmatch(3,0.,1.0*bdtl,10000.,1)
      !return
      call print_transfer_matrix
      return
      end

