def getScalingFunction(self):
    
    function_name = self.config.bound
    types = {
        'steerers': _energy_scale,
        'quads': _quad_scale,
    }

    return types[function_name]

def _energy_scale(self):
        
        commons_dict = self.config.beam.common_elements
        upstream_steerers_difference = {} #saves the difference of (steerer-common) for each steerer as a dict with the key as the steerer pv
        for common in commons_dict.keys():
            for steerer in commons_dict[common]['commonto']:
                if steerer not in self.steerers_dict.keys():
                    diff_dict = self.target_function.get_jaya(list([steerer,common]))
                    # print(diff_dict.items())
                    upstream_steerers_difference[steerer]=float(diff_dict[steerer])-float(diff_dict[common])
        
        if self.config.beam.beam_energy==None:
            print("oopsies looks like you forgot to input the beam energy, silly ;)")
            self.beam_energy = input("beam energy (kev): ")
        else:
            self.beam_energy = self.config.beam.beam_energy
        if self.config.beam.beam_chargestate == None:
            print("oopsies looks like you forgot to input the beam chragestate, silly ;)")
            self.beam_chargestate = input("beam chargestate: ")
        else:
            self.beam_chargestate = self.config.beam.beam_chargestate
        if self.config.beam.beam_mass == None:
            print("oopsies looks like you forgot to input the beam mass, silly ;)")
            self.beam_mass = input("beam mass (amu): ")
        else:
            self.beam_mass = self.config.beam.beam_mass
        
        print(f"energy: {self.beam_energy} keV")
        print(f"charge: {self.beam_chargestate}")
        print(f"mass: {self.beam_mass} amu")
        
        #find range that all steerers will need using energy
        allowed_half_range_dict={}
        safety_factor = 1.25 #set all ranges to be 20% larger than the strict bounds
        for steerer in self.steerers_dict.keys():
            steering_element = self.steerers_dict[steerer]
            if steering_element.type=="steerer":
                if 'VOL' in steerer:
                    allowed_half_range_dict[steerer] = safety_factor*self.beam_energy * 4 # 120 V allowed steering for 30keV beam, energy is in keV
                elif 'CUR' in steerer: 
                    if steering_element.upper_bound == 3:# two types of steerers have two different magnets
                        allowed_half_range_dict[steerer]=0.05*self.beam_mass/self.beam_chargestate #amps
                    elif steering_element.upper_bound == 100:
                        allowed_half_range_dict[steerer]=1.7*self.beam_mass/self.beam_chargestate #amps
                    else:
                        print(f"ERROR: steerer '{steerer}' is of an unsupported type. Only magnetic steerers with range +/-3 and +/-100 are supported.")
                        exit()
                else:
                    print(f"ERROR: steerer '{steerer}' is of an unsopported type. Not recognized as electrostatic or magnetic steerer. ")
                    exit()
            elif steering_element.type == 'quad':  #for testing on hebt3
                allowed_half_range_dict[steerer] = safety_factor*5
            elif steering_element.type == 'eb':
                if steering_element.theta == '-45*deg' or '45*deg':
                    allowed_half_range_dict[steerer] = safety_factor*23 * self.beam_energy / 30 # V/ keV

                elif steering_element.theta == '-9*deg' or '9*deg':
                    allowed_half_range_dict[steerer] = safety_factor*79 * self.beam_energy / 30 # V/ keV
            else:
                allowed_half_range_dict[steerer] = None

            try:
                if allowed_half_range_dict[steerer] >= (self.steerers_dict[steerer].upper_bound - self.steerers_dict[steerer].lower_bound):
                    print(f"Warning: scaled range of {allowed_half_range_dict[steerer]} for {steerer} is greater than half the bounds")
                    print(f"setting the allowed half range to half the bounds, please check your energy is in keV.")
                    allowed_half_range_dict[steerer]= self.steerers_dict[steerer].upper_bound - self.steerers_dict[steerer].lower_bound
            except TypeError:
                pass
            
            print(f"scaled range {steerer}: {allowed_half_range_dict[steerer]}")
        
        # find lowest that all steers in a bunch can be set to knowing the necessary range
        mean_dict = {}
        common_setting_dict={}
        if commons_dict:
            for common in commons_dict.keys():
                max_range = 0
                # find the largest allowed half range of all steerers in a bunch (probably the same)
                for steerer in commons_dict[common]['commonto']:
                    if steerer in self.steerers_dict.keys(): #avoid touching the steerers that tied to the first common
                        if allowed_half_range_dict[steerer]>max_range:
                            max_range = allowed_half_range_dict[steerer]
                # set all steerers mean to this common value
                for steerer in commons_dict[common]['commonto']:
                    if steerer in self.steerers_dict.keys():
                        mean_dict[steerer]=max_range
                common_setting_dict[common]=max_range
        else: # must be magnetic
            for steerer in self.steerers_dict:
                mean_dict[steerer]=0.0
                
        
        #define the mean_dict, range_dict, and var_dict accordingly
        var_dict = {}
        range_dict = {}
        for steerer in self.steerers_dict.keys():
            range_dict[steerer] = 2*allowed_half_range_dict[steerer]
            var_dict[steerer]= 0.5*allowed_half_range_dict[steerer] 
            # var defines the std dev of the normal dist for exploration. 
            # Useful to have 86% of points fall within the bound set
            # all points outside get clamped to the bounds, so try to reduce that
            
        #set the new points for the upstream steerers to preserve the potential difference
        upstream_steerers_settings = {}
        for steerer in upstream_steerers_difference.keys():
            for common in commons_dict.keys():
                for commonto in commons_dict[common]['commonto']:
                    if steerer in commonto:
                        upstream_steerers_settings[steerer] = common_setting_dict[common]+upstream_steerers_difference[steerer]
        
        #set the new scaled value for the commons, and the corresponding value for upstream affected steerers to preserve potential difference
        if common_setting_dict:
            self.target_function.send_measurement(common_setting_dict)
        if upstream_steerers_settings:
            self.target_function.send_measurement(upstream_steerers_settings)
        
        return mean_dict, var_dict, range_dict
        
        # 3. define the mean_dict, range_dict, and var_dict accordingly
        # 4. set all commons to their calculated points (mean_dict)
        # 5. return new dicts for optimizer

def _quad_scale(
        self,
        scale_percent: float = 0.10,   # half-band (±5 %)
        var_fraction: float = 0.02     # σ = 2 % of set-point
    ):
    """
    Return mean / var / range dicts for *all* PVs, scaling only quadrupoles.

    Any PV whose element.type != "quad" keeps its existing var / range
    values (pulled from self.var_dict / self.range_dict).  Nothing else
    about it is changed.
    """

    # Live set-points for every PV we control
    current_settings = self.target_function.read_inputs_full()
    mean_dict = {}
    var_dict = {}
    range_dict = {}

    # ----------------------------------------------------------------------
    for pv in self.steerers_dict.keys():
        element = self.steerers_dict[pv]
        val     = float(current_settings[pv])

        # ---------------------- quadrupoles --------------------------------
        if element.type == "quad":
            half_range          = abs(val) * scale_percent
            mean_dict[pv]       = val
            var_dict[pv]        = abs(val) * var_fraction
            range_dict[pv]      = 2.0 * half_range

            # optional clamp against hardware limits
            try:
                ub, lb = element.upper_bound, element.lower_bound
                if range_dict[pv] > (ub - lb):
                    range_dict[pv] = ub - lb
            except AttributeError:
                pass
            
            print(f"scaled range {pv}: {round(mean_dict[pv]-half_range,2)} to {round(mean_dict[pv]+half_range,2)}")

        # -------------------- everything else ------------------------------
        else:
            mean_dict[pv]  = val
            var_dict[pv]   = self.steerers_dict[pv].starting_var
            range_dict[pv] = self.steerers_dict[pv].upper_bound - self.steerers_dict[pv].lower_bound

    return mean_dict, var_dict, range_dict