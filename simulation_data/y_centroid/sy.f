      SUBROUTINE TSYSTEM
      COMMON/SCPARM/QSC,ISC,CMPS
      COMMON/MOM/P,BRHO,pMASS,ENERGK,GSQ,ENERGKi,charge,current
      COMMON/PRINT/IPRINT
      COMMON/SS/SX(13,6)
      COMMON/BLOC1/SGX1,SGY1,SGX2,SGY2,AGX1,AGY1,QM1,ABX1,ABY1,ABZ1,EMBX
     &1,RBX1,XCB1,YCB1,AGX2,AGY2,QM2,AGX3,AGY3,QM3,XCB2,YCB2,AGX4,AGY4,Q
     &M4,AGX5,AGY5,QM5,XCB3,YCB3,AGX6,AGY6,QM6,AGX7,AGY7,QM7,AGX8,AGY8,Q
     &M8,AGX9,AGY9,QM9

      CMPS=2.13 ! Number of cm per step, for plotting only
      wo=1.0 ! Weight aberration from optical elements

      ! Built from acc at commit: 12807ca5, commit date: 2026-04-08

      call genshift(SGX1,SGX2,SGY1,SGY2)
      call marker('HEBT2:FC1')
      call drift(24.9991,".")

      call genshift(AGX1,0.0,AGY1,0.0)
      ! Magnetic quadrupole HEBT2:Q2
      call mquad(-0.00029746*QM1,1.0,8.0,wo,'HEBT2:Q2')
      call genshift(-AGX1,0.0,-AGY1,0.0)

      call drift(54.956,".")

      call genshift(ABX1,0.0,ABY1,0.0)
      call edge(11.25,100.003,22.5,0.0,0.55,0.0,5.53,0.0,wo)
      call bend(100.003,22.5/2,0.0,'.')
      call marker('HEBT2:MB1')
      call genshift(0.,EMBX1,0.,RBX1)
      call bend(100.003,22.5/2,0.0,'.')
      call edge(11.25,100.003,22.5,0.0,0.55,0.0,5.53,0.0,wo)
      call genshift(-
     &ABX1*cos(0.392699*pi/180)+ABZ1*cos(0.392699*pi/180),0.0,-ABY1,0.0)
      call drift(-ABX1*sin(0.392699*pi/180),".")

      call drift(60.4108,".")
      ! Steerer HEBT2:XCB2
      call genshift(0.,XCB1,0.,0.)
      ! Steerer HEBT2:YCB2
      call genshift(0.,0.,0.,YCB1)
      call drift(62.4874,".")

      call genshift(AGX2,0.0,AGY2,0.0)
      ! Magnetic quadrupole HEBT2:Q3
      call mquad(-3.19953e-10*QM2**5.0 + 5.03293e-08*QM2**4.0 -
     &2.98602e-06*QM2**3.0 + 7.328e-05*QM2**2.0 + 0.0086354*QM2 +
     &0.0043,2.6,18.0,wo,'HEBT2:Q3')
      call genshift(-AGX2,0.0,-AGY2,0.0)

      call drift(34.9997,".")

      call genshift(AGX3,0.0,AGY3,0.0)
      ! Magnetic quadrupole HEBT2:Q4
      call mquad(3.19953e-10*QM3**5.0 - 5.03293e-08*QM3**4.0 +
     &2.98602e-06*QM3**3.0 - 7.328e-05*QM3**2.0 - 0.0086354*QM3 -
     &0.0043,2.6,18.0,wo,'HEBT2:Q4')
      call genshift(-AGX3,0.0,-AGY3,0.0)

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
      !call fit(1, 13, 1, 0.0, 10.0*wo, 1) ! x centroid
      call fit(1, 13, 3, 0.0, 10.0*wo, 1) ! y centroid
      ! Element of unknown type: ffc
      call marker('HEBT2:FC4')
      call drift(16.7331,".")
      ! Steerer HEBT2:XCB4
      call genshift(0.,XCB2,0.,0.)
      ! Steerer HEBT2:YCB4
      call genshift(0.,0.,0.,YCB2)
      call drift(15.3152,".")

      call genshift(AGX4,0.0,AGY4,0.0)
      ! Magnetic quadrupole HEBT2:Q5
      call mquad(3.19953e-10*QM4**5.0 - 5.03293e-08*QM4**4.0 +
     &2.98602e-06*QM4**3.0 - 7.328e-05*QM4**2.0 - 0.0086354*QM4 -
     &0.0043,2.6,18.0,wo,'HEBT2:Q5')
      call genshift(-AGX4,0.0,-AGY4,0.0)

      call drift(30.0004,".")

      call genshift(AGX5,0.0,AGY5,0.0)
      ! Magnetic quadrupole HEBT2:Q6
      call mquad(-3.19953e-10*QM5**5.0 + 5.03293e-08*QM5**4.0 -
     &2.98602e-06*QM5**3.0 + 7.328e-05*QM5**2.0 + 0.0086354*QM5 +
     &0.0043,2.6,18.0,wo,'HEBT2:Q6')
      call genshift(-AGX5,0.0,-AGY5,0.0)

      call drift(14.9495,".")
      ! Steerer HEBT2:XCB6
      call genshift(0.,XCB3,0.,0.)
      ! Steerer HEBT2:YCB6
      call genshift(0.,0.,0.,YCB3)
      call drift(15.05,".")

      call genshift(AGX6,0.0,AGY6,0.0)
      ! Magnetic quadrupole HEBT2:Q7
      call mquad(3.19953e-10*QM6**5.0 - 5.03293e-08*QM6**4.0 +
     &2.98602e-06*QM6**3.0 - 7.328e-05*QM6**2.0 - 0.0086354*QM6 -
     &0.0043,2.6,18.0,wo,'HEBT2:Q7')
      call genshift(-AGX6,0.0,-AGY6,0.0)

      call drift(29.9995,".")

      call genshift(AGX7,0.0,AGY7,0.0)
      ! Magnetic quadrupole HEBT2:Q8
      call mquad(3.19953e-10*QM7**5.0 - 5.03293e-08*QM7**4.0 +
     &2.98602e-06*QM7**3.0 - 7.328e-05*QM7**2.0 - 0.0086354*QM7 -
     &0.0043,2.6,18.0,wo,'HEBT2:Q8')
      call genshift(-AGX7,0.0,-AGY7,0.0)

      call drift(11.9384,".")
      ! HEBT2:IV8
      call drift(34.6116,".")
      ! Slit Dragon-Gas-Slit-1
      !call slit(1.5,1.5,wo,'Dragon-Gas-Slit-1')
      !call fit(1, 13, 1, 0.0, 10.0*wo, 1) ! x centroid
      call fit(1, 13, 3, 0.0, 10.0*wo, 1) ! y centroid
      call drift(20.0,".")
      ! Slit Dragon-Gas-Slit-2
      !call slit(1.3,1.3,wo,'Dragon-Gas-Slit-2')
      call drift(10.0,".")
      ! Slit Dragon-Gas-Slit-3
      !call slit(1.18,1.18,wo,'Dragon-Gas-Slit-3')
      call drift(15.4,".")
      ! Slit Dragon-Gas-Slit-4
      !call slit(0.99,0.99,wo,'Dragon-Gas-Slit-4')
      call drift(6.15,".")
      ! Slit Dragon-Gas-Slit-5
      !call slit(1.0,1.0,wo,'Dragon-Gas-Slit-5')
      call drift(15.2,".")
      ! Slit Dragon-Gas-Slit-6
      !call slit(0.8,0.8,wo,'Dragon-Gas-Slit-6')
      call drift(1.265,".")
      ! Slit Dragon-Gas-Slit-7
      !call slit(0.6,0.6,wo,'Dragon-Gas-Slit-7')
      call drift(6.435,".")
      ! mcat-set-dragon-gas
      ! MCAT-verb: set-dragon-gas
      !nlines: 4
      !call twissmatch(1,0.,bx1,10.,1)
      !call twissmatch(3,0.,bx1,10.,1)
      call waist(1,10.0,1)
      call waist(3,10.0,1)
      !call fit(1,1,6,0.0,1.,1)
      !return
      ! DRAGON-Gas
      ! endOf_hebt2_db0
      ! DRAGON-Gas
      call drift(5.435,".")
      ! Slit Dragon-Gas-Slit-8
      call slit(0.8,0.8,wo,'Dragon-Gas-Slit-8')
      !call fit(1, 13, 1, 0.0, 10.0*wo, 1) ! x centroid
      call fit(1, 13, 3, 0.0, 10.0*wo, 1) ! y centroid
      call drift(2.265,".")
      ! Slit Dragon-Gas-Slit-9
      !call slit(0.9,0.9,wo,'Dragon-Gas-Slit-9')
      call drift(15.2,".")
      ! Slit Dragon-Gas-Slit-10
      !call slit(1.18,1.18,wo,'Dragon-Gas-Slit-10')
      call drift(6.0,".")
      ! Slit Dragon-Gas-Slit-11
      !call slit(1.805,1.805,wo,'Dragon-Gas-Slit-11')
      call drift(3.5,".")
      ! Slit Dragon-Gas-Slit-12
      !call slit(1.805,1.805,wo,'Dragon-Gas-Slit-12')
      call drift(4.3,".")
      ! Slit Dragon-Gas-Slit-13
      !call slit(1.79,1.79,wo,'Dragon-Gas-Slit-13')
      call drift(7.8,".")
      ! Slit Dragon-Gas-Slit-14
      !call slit(2.06,2.06,wo,'Dragon-Gas-Slit-14')
      call drift(9.9,".")
      ! Slit Dragon-Gas-Slit-14
      !call slit(2.5,2.5,wo,'Dragon-Gas-Slit-14')
      call drift(30.5,".")
      ! Slit Dragon-Gas-Slit-14
      !call slit(3.6,3.6,wo,'Dragon-Gas-Slit-14')
      call drift(21.985,".")
      

      call genshift(AGX8,0.0,AGY8,0.0)
      ! Magnetic quadrupole DRA:Q1
      call mquad(-0.00092*QM8,5.3975,25.23,wo,'DRA:Q1')
      call genshift(-AGX8,0.0,-AGY8,0.0)

      call drift(25.6925,".")

      call genshift(AGX9,0.0,AGY9,0.0)
      ! Magnetic quadrupole DRA:Q2
      call mquad(0.00097*QM9,7.9375,33.385,wo,'DRA:Q2')
      call genshift(-AGX9,0.0,-AGY9,0.0)

      call drift(31.9955,".")
      call marker('DRA:FC1')
      call print_transfer_matrix
      return
      end
