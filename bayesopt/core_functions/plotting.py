import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch
import pandas as pd
import os
from omegaconf import OmegaConf
import pickle
import math
from PIL import Image
import cv2

class Plotting:
    def __init__(self, data):
        self.data = data

    def __getattr__(self, name):
        try:
            return getattr(self.data, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' "
                f"nor does the underlying data object of type "
                f"'{type(self.data).__name__}'"
            )

    def final_data_processing(self, train_x, train_y, width):
        df = pd.DataFrame({"train_x": train_x.tolist(), "train_y":  train_y.squeeze().tolist()})
        df.to_csv("training_set.csv")
        
        optimal_settings = OmegaConf.create({name: value.item() for name, value in zip(self.steerers_dict.keys(), self.optimal_input.squeeze())})
        optimal_settings['transmission'] = self.max
        with open('optimal_input.yaml', 'w') as f:
            f.write(OmegaConf.to_yaml(optimal_settings))

        self._save_all(train_x, train_y, width)

        self.memory = np.array(self.memory, dtype=object)
        # self.transmission_coeff = 100/self.config.beam.initial_transmission

        self._progress()
        
        if 'sim' in self.config:
            plt.clf()
            best_x = self.optimal_input.unsqueeze(-2)
            cv2.imwrite('best_beamline.png', self.target_function.beamline.render(show_steerer_pos=True, show_transmission=True))   
            # np.savez('best_beamline.png', self.target_function.beamline.render(show_steerer_pos=True, show_transmission=True))
            plt.close()
        
        if 'VOL' in list(self.steerers_dict.keys())[0]:
            self.y_label = "Voltage (V)"
        else:
            self.y_label = "Current (A)"

        torch.save(self.model.state_dict(), 'model_state.pth')
        if self.make_plots:
            print('Making Plots')
            try: 
                self._gaussian()
            except Exception as e:
                print('plotting gaussian failed with error: ', e)
            try:
                self._kernel()
            except Exception as e:
                print('plotting kernel failed with error: ', e)
            try:
                self._length_scale(width)
            except Exception as e:
                print('plotting length scale failed with error: ', e)
            try: 
                self._x_explored(train_x)
            except Exception as e:
                print('plotting x_explored failed with error: ', e)
                
            # response = input('plot scan vs fit (y/n)?')
            # if response == 'y' or response == 'Y':
            if self.config.fit_vs_scan:
                try:
                    self._fit_vs_scan()
                except Exception as e:
                    print('plotting scan vs fit failed with error: ', e)
                             
        print("Initial Current", self.memory[0], "pA")
        print("Max Current: ", self.max, "pA")
        print("Optimal input: ", self.optimal_input)

        self.x, self.y, self.widths = train_x, train_y, width

        if 'sim' in self.config:
            self._beamprofile()
            if self.config.save_gif:
                self._save_gif()

    def _get_figure_dimensions(self):
        """
        Calculate the dimensions for arranging subplots in a figure.

        This function calculates the number of rows and columns for arranging subplots in a figure
        based on the number of steerers defined in the 'steerers_dict'.

        Returns:
        r (int): Number of rows for subplot arrangement.
        c (int): Number of columns for subplot arrangement.
        """
        num = len(list(self.steerers_dict.keys()))
        r = 0
        while (r+1)**2 < num:
            r += 1 

        c = math.ceil(num/r)
        return r, c

    def _reorder(self, x, dim):
        OPTIMUM = self.optimal_input[dim].item()

        # find the index of the point closest to the optimum
        k = torch.argmin(torch.abs(torch.sub(x, OPTIMUM)))
        
        # which side of the optimum is the point on?
        # True if left, False if right
        LHS = (2*k+1 <= len(x)) # plus one because of the way python indexes
        new_x = [x[k]]
        
        if LHS:
            for i in range(1, k+1):
                new_x.append(x[k+i])
                new_x.append(x[k-i])
            
            a = 2*k+1
            if a!=0:
                new_x.extend(x[a:])

        else:
            a = min(len(x) - k, k)
            for i in range(1, a):
                new_x.append(x[k+i])
                new_x.append(x[k-i])
            
            if a!=0:
                new_x.extend(x[:(k-a+1)])
        
        return torch.tensor(new_x)

    def _fix_subplot_dimensions(self,r,c, axs):
        if r == 1 and c > 1:
            axs = [axs]  # Convert to a list to make it consistent with the previous structure
        
        elif c == 1 and r > 1:
            axs = [[a] for a in axs]  # Convert to a 2D list to make it consistent with the previous structure
        
        elif c ==1 and r ==1:
            axs = [[axs]]  # Convert to a 2D list to make it consistent with the previous structure
        
        return axs

    def _save_gif(self):
        def get_combined_img(beam_img, blurb):
            fontsize = 1
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            max_width = int(max([cv2.getTextSize(line, font, fontsize, font_thickness)[0][0] for line in blurb])*1.2)

            textbox = np.ones((beam_img.size[0], max_width,3), dtype='uint8')*255
            i = 0
            for line in blurb:
                textsize = cv2.getTextSize(line, font, fontsize, font_thickness)[0]
                gap = int((beam_img.size[1] - len(blurb)*textsize[1])/(len(blurb)+1))
                y = textsize[1]*2 + i * (gap+textsize[1])
                x = int(fontsize*10)

                cv2.putText(textbox, line, (x, y), font,
                            fontsize, 
                            (0,0,0), 
                            font_thickness, 
                            lineType = cv2.LINE_AA)
                i +=1
            textbox = Image.fromarray(textbox)
            combined_img = Image.new('RGB',(beam_img.size[0]+textbox.size[0], beam_img.size[1]), (250,250,250))
            combined_img.paste(beam_img,(0,0))
            combined_img.paste(textbox, (beam_img.size[0],0) )
            return combined_img
        
        frames = []
        for state in self.target_function.beamline_states:
            beam_img = Image.fromarray(state['img']).convert("RGB")
            # beam_img = Image.fromarray(state['img'][:,:,::-1].astype('uint8')).convert("RGB")
            step = state['step']
            transmission = round(state["transmission"],2)
            action_dict = state["action"]
            action_dict = {k: round(v.item(), 2) for k,v in action_dict.items()}
            blurb = [f"STEP: {step}"] + str(action_dict).replace("{","").replace("}","").split(", ") + [f"TRANSMISSION: {transmission}"]
                
            combined_img = get_combined_img(beam_img, blurb) 
            frames.append(combined_img)
        
        for i in range(5):
            state = self.target_function.best_beamline_state
            beam_img = Image.fromarray(state['img'][:,:,::-1].astype('uint8')).convert("RGB")
            transmission = round(state["transmission"], 2)
            action_dict = state["action"]
            action_dict = {k: round(v.item(), 2) for k,v in action_dict.items()}
            blurb = [f"FINAL SETTING"] + str(action_dict).replace("{","").replace("}","").split(", ") + [f"TRANSMISSION: {transmission}"]
                
            combined_img = get_combined_img(beam_img, blurb) 
            frames.append(combined_img)
        
        print('saving gif of run ...')
        frames[0].save("bo_tuning_results.gif", format="GIF", append_images=frames, duration=300,loop=True, _save_all=True)

    def _progress(self, fig_title=""):
        """
        Plots the progress of the optimization process.

        This function reads a training dataset from 'training_set.csv' and plots the progress of optimization
        over iterations. It categorizes the data into initial sampling, optimizing, and zero beta points.
        The resulting plot shows these categories with different colors and markers.

        Args:
        fig_title (str): Title for the plot (optional).

        Returns:
        None
        """
        sns.set_theme(style="darkgrid")  # or use plt.style.use('seaborn-whitegrid')
        
        # Read the training dataset from 'training_set.csv'
        train_data = pd.read_csv('training_set.csv')

        # Extract the 'train_y' column and convert it to a list of float values
        y_Data = list(map(lambda x: float(x), train_data['train_y']))[:-1]

        # Create a figure for the plot
        plt.figure(figsize=(8,4))
        
        # Extract configuration parameters
        sp = self.initial_points
        it = self.config.iterations

        # Categorize the data into initial sampling, optimizing, and zero beta points
        initial_sample_points = y_Data[:sp]
        if len(y_Data) > sp+it:
            optimizing_points = y_Data[sp:sp+it]
            zero_beta_points = y_Data[sp+it:]
        else:
            optimizing_points = y_Data[sp:]
            zero_beta_points = []

        # Scale the data using the transmission coefficient
        scaled_initial = [point * self.transmission_coeff for point in initial_sample_points]
        scaled_optimizing = [point * self.transmission_coeff for point in optimizing_points]
            
        plt.clf()

        # Plot the data with different colors and markers
        plt.plot(np.arange(len(initial_sample_points)), scaled_initial, color='red', marker='o', label='initial sampling')
        plt.plot(np.arange(len(optimizing_points))+len(initial_sample_points), scaled_optimizing, color='blue', marker='o', label='optimzing')
        
        # Set the plot title based on the configuration
        if 'sim' in self.config:
            plt.title('Progress for '+os.getcwd().split('/')[-1]+f', Beta={self.config.beta}')
        elif 'beam' in self.config:
            plt.title('Progress for '+  next(iter(self.steerers_dict)) + ' to ' +self.config.beam.measurement_device.pv +f', Beta={self.config.beta}')
        
        # Set labels for the x and y axes, set legend, set layout
        plt.xlabel("Iteration",fontsize=12)
        plt.ylabel(self.ylabel,fontsize=12)
        plt.legend(bbox_to_anchor=(0,1.06,1,0.2), loc="lower left", mode="expand", ncol=3)
        plt.tight_layout()
        plt.savefig("Progress.png", bbox_inches='tight', pad_inches=0, dpi=300)  # Adjust settings as needed
        plt.close()
            
    def _beamprofile(self, fig_title=""):
        print("plotting beam profile")
        plt.clf()
        x_Data = range(1,len(self.memory))
        plt.figure(figsize=(8,4))
        # plt.plot(x_Data, self.sizex, color="blue", marker='.', label=f'x-env at {self.target_function.beamline.profile_device}') 
        # plt.plot(x_Data, self.sizey, color="green", marker='.', label=f'y-env at {self.target_function.beamline.profile_device}') 
        # plt.plot(x_Data, self.cx, "--", color="blue", label=f'x-cm at {self.target_function.beamline.profile_device}') 
        # plt.plot(x_Data, self.cy, "--", color="green", marker='.', label=f'y-cm at {self.target_function.beamline.profile_device}') 
        plt.xlabel("Iteration",fontsize=12)
        plt.ylabel("Size/cm",fontsize=12)
        plt.legend(fancybox=True)
        plt.title(fig_title)
        plt.savefig("profile.png", bbox_inches='tight', pad_inches=0, dpi=300)  # Adjust settings as needed
        plt.close()

    def _gaussian(self): 
        """
        Plots Gaussian distributions.

        This function generates plots of Gaussian distributions for different steerers. It plots the mean
        and confidence intervals of the Gaussian distributions.

        Returns:
        None
        """
        print('plotting gaussian')
        model = self.model

        with torch.no_grad():

            plt.clf()

            # Number of points for plotting
            pts = 50
            test_x = self.optimal_input.repeat(pts, 1)
            torch.set_default_dtype(torch.float64)

            # Get the names of steerers from the 'steerers_dict'
            steerer_names = list(self.steerers_dict.keys())

            # Calculate the number of rows and columns for subplots
            r,c = self._get_figure_dimensions()

            # Create a figure and a grid of subplot
            fig, axs = plt.subplots(r, c, sharex=False, sharey=True, figsize=(16, 13))

            # Fix subplot dimensions
            ax = self._fix_subplot_dimensions(r,c,axs)
        

            dim = 0

            for i in range(r):
                for j in range(c):
                    if self.dim==dim:
                        break
                    

                    vector = torch.linspace(self.bounds[0, dim], self.bounds[1, dim], pts)
                    
                    # Create input data with varying values for the current steerer
                    x = test_x.clone()
                    x[:, dim] = vector
                    # Get posterior from the model
                    posterior = model.posterior(x)

                    # Extract mean and confidence intervals
                    mean = posterior.mean
                    mean = torch.squeeze(mean, 0)
                    lb, ub = posterior.confidence_region()
                    lb, ub = torch.squeeze(lb, 0), torch.squeeze(ub, 0)

                    # Sort the vector for plotting
                    vector, indices = torch.sort(vector)
                    y = mean[indices]

                    # Plot mean and confidence intervals
                    ax[i][j].plot(vector, y, label = "Mean", color="blue")
                    ax[i][j].fill_between(vector, lb, ub, color = "blue",alpha=0.2)

                    # Add legend, title, and increment steerer index
                    ax[i][j].legend(loc="lower left")
                    ax[i][j].set_title(steerer_names[dim])
                    dim += 1
            
            # Set labels for the x and y axes
            ax[r-1][0].set_xlabel(self.xlabel,fontsize=20)
            ax[0][0].set_ylabel(self.ylabel,fontsize=20)
            
            plt.savefig("gaussian.png", bbox_inches='tight', pad_inches=0, dpi=300)  # Adjust settings as needed
            plt.close()
            
    def _length_scale(self, width):
        """
        Plots length scales for steerers.

        This function generates plots that show the length scales for different steerers based on the provided 'width' data.

        Args:
        width (list): A list of lists representing the width data for each steerer.

        Returns:
        None
        """

        print("plotting length scale")
        plt.clf()

        # Get the names of steerers from the 'steerers_dict'
        steerer_names = list(self.steerers_dict.keys())

        # Convert width to a tensor and squeeze it
        width = torch.tensor(width).squeeze()
        width = width.tolist()
        
        # Calculate the number of rows and columns for subplots
        r,c = self._get_figure_dimensions()

        # Create a figure and a grid of subplots
        fig, ax = plt.subplots(r, c, sharex=False, sharey=True, figsize=(15, 13))

        # Fix subplot dimensions
        axs = self._fix_subplot_dimensions(r,c,ax)
        
        
        k = 0
        for i in range(r):
            for j in range(c):
                if self.dim==k:
                    break

                # Extract width data for the current steerer & gen x-axis data
                yData = [item[k] for item in width]
                xData = range(len(yData))
                
                # Plot width data
                axs[i][j].plot(xData, yData, 'bo')
                axs[i][j].plot(xData, yData, color='blue')
                axs[i][j].set_title(steerer_names[k],fontsize=15)
                k+=1
        
        # Set title, labels, and draw the figure canvas
        plt.suptitle('length scale', fontsize=25)
        axs[r-1][0].set_xlabel(self.xlabel,fontsize=20)
        axs[0][0].set_ylabel('Length Scale',fontsize=20)
        plt.savefig("length_scales_compare.png", bbox_inches='tight', pad_inches=0, dpi=300)  # Adjust settings as needed
        plt.close()

    def _covar_matrix(self,model, train_x, key=0):
        """
        Plots the covariance matrix associated with a Gaussian Process model.

        This function computes and visualizes the covariance matrix of the GP model, which represents the relationships
        between data points in the input space.

        Args:
        model (GPyTorch Model): The Gaussian Process model.
        train_x (Tensor): The training input data.
        key (int): An optional key or identifier for the plot.

        Returns:
        None
        """

        print('plotting covariance matrix')

        # Transpose the training input data for compatibility with the model
        x = torch.transpose(train_x, 0, 1)

        # Compute the covariance matrix using the model's covar_module
        covar = model.covar_module(x)
        covar = covar.evaluate()
        covar = covar.detach()

        # Create a heatmap of the covariance matrix using a color map ('hot' in this case)
        plt.imshow(covar, cmap='hot', interpolation='nearest')
        plt.title(f"Iteration {key}")
        plt.savefig(f"covariance_matrix_{key}.png")

    def _fit_vs_scan(self, figtitle=''):
        """
        Plots fit vs. scan data for optimization progress.

        This function generates plots that compare fit data with scan data during the optimization process.
        It creates subplots for different steerers, showing the mean, confidence intervals, and observed data points.

        Args:
        figtitle (str): Title for the main figure (optional).

        Returns:
        None
        """

        print("plotting fit vs scan")

        model = self.model

        with torch.no_grad():

            plt.clf()

            # Number of points for plotting
            pts = 30
            test_x = self.optimal_input.repeat(pts, 1)

            torch.set_default_dtype(torch.float64)
            steerer_names = list(self.steerers_dict.keys())
            
            r,c = self._get_figure_dimensions()
        
            f, ax = plt.subplots(r,c, sharex = False, sharey=True, figsize=(16,13))
            ax = self._fix_subplot_dimensions(r,c, ax)

            dim = 0
            xs, results, preds = [], [], []

            scan_data = {}
            for i in range(r):
                for j in range(c):
                    # if dim == 0 or dim == 1:
                    #     dim += 1
                    #     continue
                    
                    if self.dim==dim:
                        break
                    # print(f'r: {r}    c:{c}     dim:{dim}    self.dim:{self.dim}')
                    # print(f'steerer dict: {steerer_names}')
                    # print(f'scanning {pts} points for {steerer_names[dim]}')
                    # pass the full range into the posterior model
                    test_x = self.optimal_input.repeat(pts, 1)
                    vector = torch.linspace(self.bounds[0, dim], self.bounds[1, dim], pts)
                    x = test_x.clone()
                    x[:, dim] = vector
                
                    posterior = model.posterior(x)

                    mean = posterior.mean
                    mean = torch.squeeze(mean)
                    lb, ub = posterior.confidence_region()
                    lb, ub = torch.squeeze(lb), torch.squeeze(ub)
                    
                    #constrain the range to the mean +/- 2 std
                    x_i = torch.argmax(mean)
                    # lb_xData = 0.8*mean_xData
                    # ub_xData = 1.2*mean_xData
                    
                    # k1, k2 = torch.argmin(torch.abs(torch.sub(mean, lb_xData))), torch.argmin(torch.abs(torch.sub(mean, ub_xData)))
                    
                    lb_xData, ub_xData = vector[max(x_i-3, 0)], vector[min(x_i+3, pts-1)]

                    # pass the constrained range into the posterior model
        
                    pts=self.config.fit_vs_scan_points
                    test_x = self.optimal_input.repeat(pts, 1)
                    vector = torch.linspace(lb_xData, ub_xData, pts)
                    vector__reordered = self._reorder(vector, dim)
                    
                    x = test_x.clone()
                    x__reordered = test_x.clone()
                    x[:, dim] = vector
                    x__reordered[:, dim] = vector__reordered

                    posterior = model.posterior(x)

                    mean = posterior.mean
                    mean = torch.squeeze(mean, 0)
                    lb, ub = posterior.confidence_region()
                    lb, ub = torch.squeeze(lb, 0), torch.squeeze(ub, 0)
                    
                    print('here in sequence ' + str(dim))
                    yData = self.target_function(x__reordered)[0]
                    #print(yData)

                    #yErr = yStd (sexually transmitted diseases stay away :))
                    yStd = torch.full_like(yData, np.sqrt(self.y_variance))
                    #print(yStd)
                    #print(yStd.tolist())
                    finalyErr = []
                    finalyData = []
                    for obj in yStd:
                        finalyErr.append(obj[0])
                    for obj in yData:
                        finalyData.append(obj[0])
                    #print("I made it here")
                        # print(finalyErr)
                    scan_data[steerer_names[dim]]={'x':x__reordered, 'y':yData.tolist()}
                    with open('scan_data.pkl', 'wb') as f:
                        pickle.dump(scan_data, f)

                    #print("made it here pt 2")
                    # print(yData)
                    # print(yData.tolist())
                    ax[i][j].plot(vector, mean , label = "Mean", color="blue")
                    ax[i][j].fill_between(vector, lb, ub, color = "blue",alpha=0.2)
                    ax[i][j].plot(vector__reordered, yData, "ko", label = "Observation",markersize=3.0)
                    ax[i][j].axvline(x=self.optimal_input[dim], label = "Best", color="#B22222", linestyle='--', linewidth=0.75)
                    # ax[i][j].errorbar(vector__reordered, finalyData, finalyErr, fmt='ko', markersize=1.0)
                    ax[i][j].legend(loc="lower left")
                    #ax[i][j].set_title(steerer_names[dim])
                    ax[i][j].set_xlabel(steerer_names[dim])
                    min_y_value = min(min(mean), min(lb), min(ub), min(yData))
                    # print(min_y_value)
                    # print(max(0,min_y_value))
                    # ax[i][j].set_ylim(max(0, min_y_value),)
                    ax[i][j].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, pos: f'{y * self.transmission_coeff:.0f}' ))

                    xs.append(vector)
                    results.append(yData)
                    dim += 1
                    df = pd.DataFrame({"x":xs, "y":results})
                    df.to_csv("scan_data.csv")
                    # plt.savefig(f"fit_vs_scan.png")

            plt.savefig("fit_vs_scan.png", bbox_inches='tight', pad_inches=0, dpi=300)  # Adjust settings as needed        
            plt.clf()
            plt.close()
                    
    def _save_all(self, train_x, train_y, widths):
        steerer_names = list(self.steerers_dict.keys())
        
        width = []
        if self.dim==1:
            widths = widths.unsqueeze(-1)
            
        for dim in range(self.dim):
            a = [item[dim] for item in widths]
            width.append(a)

        ls_dict = dict(zip(steerer_names, width))

        ls_df = pd.DataFrame(ls_dict)
        ls_df.to_csv("length_scales.csv")

        x = train_x.tolist()
        x.append(steerer_names)
        y = train_y.squeeze().tolist()
        y.append(y[-1])
        
        self.x = x
        self.y = y

        df = pd.DataFrame({"train_x": x, "train_y": y})
        df.to_csv("training_set.csv")

    def _x_explored(self, train_x):
        print("plotting x explored")
        steerer_names = list(self.steerers_dict.keys())

        if 'sim' in self.config:
            train_x = train_x*1000 # convert to mrad
            self.xlabel = 'Steerer Value (mrad)'
        
        r,c = self._get_figure_dimensions()
        fig, axs = plt.subplots(r, c, sharex=False, sharey=False, figsize=(20, 13))

        axs = self._fix_subplot_dimensions(r,c,axs)

        # print(f"r: {r}")
        # print(f"c: {c}")   

        k = 0
        for i in range(r):
            for j in range(c):
                if k == self.dim:
                    break
                yData = [item[k] for item in train_x]
                xData = range(len(yData))

                axs[i][j].plot(xData, yData, 'bo')
                axs[i][j].plot(xData, yData, color='blue')
                axs[i][j].set_title(steerer_names[k],fontsize=15)
                k+=1
        
        axs[0][0].set_ylabel(self.xlabel, fontsize=20)
        axs[r-1][0].set_xlabel("Runs", fontsize=20)
        plt.savefig("x_explored.png", bbox_inches='tight', pad_inches=0, dpi=300)  # Adjust settings as needed
        plt.close()


    def _kernel(self):
        print("plotting kernel")
        # Create a Matern kernel object with nu=2.5 and ard_num_dims=1
        kernel = self.model.covar_module.base_kernel
        
        steerer_names = list(self.steerers_dict.keys())
        r, c = self._get_figure_dimensions()
        fig, axs = plt.subplots(r, c, sharex=False, sharey=True, figsize=(16, 13))

        axs = self._fix_subplot_dimensions(r,c,axs)
        r, c = self._get_figure_dimensions()
        fig, axs = plt.subplots(r, c, sharex=False, sharey=True, figsize=(16, 13))

        axs = self._fix_subplot_dimensions(r,c,axs)
        
        steerer_names = list(self.steerers_dict.keys())
        dim = 0

        pts = 50
        test_x = self.optimal_input.repeat(pts, 1)

        torch.set_default_dtype(torch.float64)

        for i in range(r):
            for j in range(c):

                if self.dim==dim:
                    break
                
                vector = torch.linspace(self.bounds[0, dim], self.bounds[1, dim], pts)
                x = test_x.clone()
                x[:, dim] = vector

                # Compute the kernel matrix for the grid of input points
                cov = kernel(x).evaluate().squeeze().detach().numpy()

                data = np.random.multivariate_normal(vector, cov, 1000)

                # Compute the eigenvalues and eigenvectors of the covariance matrix
                eigenvalues, eigenvectors = np.linalg.eig(cov)

                # Get the index of the largest eigenvalue
                largest_eigenvalue_index = np.argmax(eigenvalues)

                # Get the corresponding eigenvector
                largest_eigenvector = eigenvectors[:, largest_eigenvalue_index]

                # Compute the angle between the largest eigenvector and the x-axis
                angle = np.arctan2(np.real(largest_eigenvector[1]), np.real(largest_eigenvector[0]))

                # Compute the standard deviation in each direction
                sigma_x = np.sqrt(eigenvalues[largest_eigenvalue_index])
                sigma_y = np.sqrt(eigenvalues[1 - largest_eigenvalue_index])

                # Create an ellipse object with the desired properties
                ellipse = matplotlib.patches.Ellipse(xy=vector, width=2 * sigma_x, height=2 * sigma_y, angle=np.degrees(angle))

                axs[i][j].plot(data[:, 0], data[:, 1], "bo" , alpha=0.2)
                axs[i][j].set_title(steerer_names[dim])
                axs[i][j].add_artist(ellipse)
                
                dim += 1
        
        plt.suptitle("Kernel Density")
        axs[r-1][0].set_xlabel(self.xlabel,fontsize=20)
        axs[0][0].set_ylabel("K(x, x*)", fontsize=20)
        
        plt.savefig("kernel.png")
