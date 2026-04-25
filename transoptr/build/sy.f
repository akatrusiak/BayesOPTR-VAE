      SUBROUTINE TSYSTEM
      COMMON/SCPARM/QSC,ISC,CMPS
      COMMON/MOM/P,BRHO,pMASS,ENERGK,GSQ,ENERGKi,charge,current
      COMMON/PRINT/IPRINT
      COMMON/SS/SX(13,6)
      COMMON/BLOC1/QM1,QM2,QM3,QM4,QM5,QM6,QM7,QM8,QM9

      CMPS=2.13 ! Number of cm per step, for plotting only
      wo=1.0 ! Weight aberration from optical elements

      ! Built from acc at commit: 142dba29, commit date: 2024-05-27

      call marker('HEBT2:XSLIT1')
      call marker('HEBT2:FC1')
      call drift(24.9991,".")
      ! Magnetic quadrupole HEBT2:Q2
      call mquad(-0.00029746*QM1,1.0,8.0,wo,'HEBT2:Q2')
      call drift(54.956,".")
      ! Magnetic dipole HEBT2:MB1
      call edge(11.25,100.003,22.5,0.0,0.55,0.0,5.53,0.0,wo)
      call bend(100.003,22.5,0.0,'HEBT2:MB1')
      call edge(11.25,100.003,22.5,0.0,0.55,0.0,5.53,0.0,wo)
      call drift(60.4108,".")
      ! HEBT2:XCB2
      ! HEBT2:YCB2
      call drift(62.4874,".")
      ! Magnetic quadrupole HEBT2:Q3
      call mquad(-3.19953e-10*QM2**5.0 + 5.03293e-08*QM2**4.0 -
     &2.98602e-06*QM2**3.0 + 7.328e-05*QM2**2.0 + 0.0086354*QM2 +
     &0.0043,2.6,18.0,wo,'HEBT2:Q3')
      call drift(34.9997,".")
      ! Magnetic quadrupole HEBT2:Q4
      call mquad(3.19953e-10*QM3**5.0 - 5.03293e-08*QM3**4.0 +
     &2.98602e-06*QM3**3.0 - 7.328e-05*QM3**2.0 - 0.0086354*QM3 -
     &0.0043,2.6,18.0,wo,'HEBT2:Q4')
      call drift(75.9752,".")
      ! mcat-set-hebt2-rpm1-2
      ! MCAT-verb: set-hebt2-rpm1-focus
      !nlines: 4
      !call twissmatch(1,0.,bx1,10.,1)
      !call twissmatch(3,0.,bx1,10.,1)
      !call fit(1,1,6,0.0,1.,1)
      !return
      call marker('HEBT2:RPM4')
      call drift(2.9751,".")
      ! Element of unknown type: ffc
      call marker('HEBT2:FC4')
      call drift(16.7331,".")
      ! HEBT2:XCB4
      ! HEBT2:YCB4
      call drift(15.3152,".")
      ! Magnetic quadrupole HEBT2:Q5
      call mquad(3.19953e-10*QM4**5.0 - 5.03293e-08*QM4**4.0 +
     &2.98602e-06*QM4**3.0 - 7.328e-05*QM4**2.0 - 0.0086354*QM4 -
     &0.0043,2.6,18.0,wo,'HEBT2:Q5')
      call drift(30.0004,".")
      ! Magnetic quadrupole HEBT2:Q6
      call mquad(-3.19953e-10*QM5**5.0 + 5.03293e-08*QM5**4.0 -
     &2.98602e-06*QM5**3.0 + 7.328e-05*QM5**2.0 + 0.0086354*QM5 +
     &0.0043,2.6,18.0,wo,'HEBT2:Q6')
      call drift(14.9495,".")
      ! HEBT2:XCB6
      ! HEBT2:YCB6
      call drift(15.05,".")
      ! Magnetic quadrupole HEBT2:Q7
      call mquad(3.19953e-10*QM6**5.0 - 5.03293e-08*QM6**4.0 +
     &2.98602e-06*QM6**3.0 - 7.328e-05*QM6**2.0 - 0.0086354*QM6 -
     &0.0043,2.6,18.0,wo,'HEBT2:Q7')
      call drift(29.9995,".")
      ! Magnetic quadrupole HEBT2:Q8
      call mquad(3.19953e-10*QM7**5.0 - 5.03293e-08*QM7**4.0 +
     &2.98602e-06*QM7**3.0 - 7.328e-05*QM7**2.0 - 0.0086354*QM7 -
     &0.0043,2.6,18.0,wo,'HEBT2:Q8')
      call drift(11.9384,".")
      ! HEBT2:IV8
      call drift(109.062,".")
      ! mcat-set-dragon-gas
      ! MCAT-verb: set-dragon-gas
      !nlines: 4
      !call twissmatch(1,0.,bx1,10.,1)
      !call twissmatch(3,0.,bx1,10.,1)
      !call fit(1,1,6,0.0,1.,1)
      !return
      ! DRAGON-Gas
      ! endOf_hebt2_db0
      ! DRAGON-Gas
      call drift(106.885,".")
      ! Magnetic quadrupole DRA:Q1
      call mquad(3.19953e-10*QM8**5.0 - 5.03293e-08*QM8**4.0 +
     &2.98602e-06*QM8**3.0 - 7.328e-05*QM8**2.0 - 0.0086354*QM8 -
     &0.0043,5.3975,25.23,wo,'DRA:Q1')
      call drift(25.6925,".")
      ! Magnetic quadrupole DRA:Q2
      call mquad(-3.19953e-10*QM9**5.0 + 5.03293e-08*QM9**4.0 -
     &2.98602e-06*QM9**3.0 + 7.328e-05*QM9**2.0 + 0.0086354*QM9 +
     &0.0043,7.9375,33.385,wo,'DRA:Q2')
      call drift(31.9955,".")
      call marker('DRA:FC1')
      call print_transfer_matrix
      return
      end
