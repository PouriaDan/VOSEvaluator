import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import glob
import os

from evaluation_tools import utils

from PIL import Image
from ipywidgets import interact, fixed
import ipywidgets as widgets
    
def make_box_layout():
     return widgets.Layout(
         display='flex',
         flex_flow='column',
         border='solid 3px black',
         margin='10px 10px 10px 10px',
         padding='5px 5px 5px 5px',
         align_items='center',
     )


class ModelAnalysis(widgets.VBox):
    def __init__(self, evaulator):
        super().__init__()
        output = widgets.Output()
        self.dataloader = evaulator.dataloader
        self.metric_res = evaulator.metrics_res
        self.seqs = list(self.dataloader.get_sequences())
        
        with output:
            self.fig, self.axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(10, 6.5))
            plt.show(self.fig)

        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.toolbar_visible = True
        self.fig.canvas.toolbar_position = 'bottom'
        for ax in self.axes[0,:]:
            ax.set_axis_off()
        
        self.selected_seq = self.seqs[0]
        
        self.int_slider = widgets.IntSlider( 
            min=0, 
            step=1, 
            description='Frame'
        )
        
        self.objects_selector = widgets.SelectMultiple(
            description='Objects')
        
        self.initiate_plot()
        
        sequence_selector = widgets.Dropdown(
            options=self.seqs, 
            description='Sequence')
        
        controls = widgets.VBox([
            widgets.Label('Control Panel'),
            widgets.HBox([sequence_selector,
                            self.int_slider, 
                            self.objects_selector]),
        ])
        
        out_box = widgets.VBox([ widgets.Label('Monitoring Dashboard'), output])

        controls.layout = make_box_layout()
        out_box.layout = make_box_layout()
        
        sequence_selector.observe(self.change_sequence, 'value')
        self.int_slider.observe(self.update_cursor, 'value')
        self.objects_selector.observe(self.update_objects, 'value')
            
        self.children = [controls, out_box]
        
    def initiate_plot(self):
        cmap = cm.get_cmap('jet')
        self.Js = self.metric_res['J']['Per_seq_object_frame'][self.selected_seq]
        self.Fs = self.metric_res['F']['Per_seq_object_frame'][self.selected_seq]
        self.object_indices = np.array(list(self.Js.keys()))
        self.num_objects = np.max(self.object_indices)
        self.img_array, self.an_array , self.pr_array  = list(self.dataloader.get_contents(self.selected_seq))
        self.num_frames = len(self.img_array)
        
        self.frame_index = 0
        self.objects_to_plot = self.object_indices.copy()
     
        out_an, out_pr = self.get_images()
        self.annotation_img = self.axes[0,0].imshow(out_an)
        self.axes[0,0].set(title=f"Ground Truth")
        self.prediction_img = self.axes[0,1].imshow(out_pr)
        self.axes[0,1].set(title=f"Model's Output")
        
        self.clear_axis()
        
        self.j_lines = []
        self.f_lines = []
        for i, o in enumerate(self.object_indices):
            j_line, = self.axes[1,0].plot(np.arange(1,self.num_frames),self.Js[o], 
                                          label=f"object {o}", 
                                          color=cmap(o/self.num_objects))
            f_line, = self.axes[1,1].plot(np.arange(1,self.num_frames),self.Fs[o], 
                                          label=f"object {o}", 
                                          color=cmap(o/self.num_objects))
            self.axes[1,0].set_ylabel("J")
            self.axes[1,1].set_ylabel("F")
            self.axes[1,0].legend()
            self.axes[1,1].legend()
            self.j_lines.append(j_line)
            self.f_lines.append(f_line)
        
        self.j_cursor = self.axes[1,0].axvline(self.frame_index, color='black', ls='--')
        self.f_cursor = self.axes[1,1].axvline(self.frame_index, color='black', ls='--')
        self.int_slider.value=0
        self.int_slider.max = self.num_frames-1
        self.objects_selector.options = self.object_indices
        self.objects_selector.value = tuple(self.object_indices)
        
    def clear_axis(self):
        self.axes[1,0].clear()
        self.axes[1,1].clear()
        if hasattr(self, 'j_lines'):
            for i in range(len(self.j_lines)):
                self.j_lines.pop(0).remove()
                self.f_lines.pop(0).remove()
        if hasattr(self, 'j_cursor'):
            self.j_cursor.remove()
            self.f_cursor.remove()
        
    def change_sequence(self, change):
        self.selected_seq = change.new
        self.initiate_plot()
        
    def update_cursor(self, change):
        self.frame_index = change.new
        out_an, out_pr = self.get_images()
        self.annotation_img.set_data(out_an)
        self.prediction_img.set_data(out_pr)
        self.j_cursor.set_data([self.frame_index,self.frame_index],[0,1])
        self.f_cursor.set_data([self.frame_index,self.frame_index],[0,1])
        
    def update_objects(self, change):
        self.objects_to_plot = change.new
        out_an, out_pr = self.get_images()
        self.annotation_img.set_data(out_an)
        self.prediction_img.set_data(out_pr)
        for i, o in enumerate(self.object_indices):
            if o in self.objects_to_plot:
                self.j_lines[i].set_ydata(self.Js[o])
                self.f_lines[i].set_ydata(self.Fs[o])
            else: 
                self.j_lines[i].set_ydata(None)
                self.f_lines[i].set_ydata(None)
        
    def get_images(self):
        annot = self.an_array[self.frame_index].copy()
        annot[~np.isin(annot, self.objects_to_plot)]=0
        pred = self.pr_array[self.frame_index].copy()
        pred[~np.isin(pred, self.objects_to_plot)]=0
        out_an = utils.merge_images(self.img_array[self.frame_index], 0.4, 
                                    utils.mask_to_seg(annot, self.num_objects), .6)
        out_pr = utils.merge_images(self.img_array[self.frame_index] ,0.4, 
                                    utils.mask_to_seg(pred, self.num_objects), .6)
        return out_an, out_pr