# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 22:44:41 2018

@author: tgill

run script to display matplotlib interface for track creation (in interactive console)
When done, run save_track(filename) to save your track.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from time import time

x1_pts = []
y1_pts = []

x2_pts = []
y2_pts = []

x_temp = []
y_temp = []

fig, ax = plt.subplots()

plt.xlim((0, 1))
plt.ylim((0, 1))

line1, = ax.plot(x1_pts, y1_pts, marker='o')
line2, = ax.plot(x2_pts, y2_pts, marker='o')
pt_temp, = ax.plot(x_temp, y_temp, marker='o')


i=0
boo = False
start = 0

def onpick(event):
    global i
    m_x, m_y = event.x, event.y
    x, y = ax.transData.inverted().transform([m_x, m_y])
    if event.button==1 and i%2==0:
        x1_pts.append(x)
        y1_pts.append(y)
        line1.set_xdata(x1_pts)
        line1.set_ydata(y1_pts)
        i = i+1
    elif event.button==3 and i%2==1:
        x2_pts.append(x)
        y2_pts.append(y)
        line2.set_xdata(x2_pts)
        line2.set_ydata(y2_pts)
        i = i+1
    fig.canvas.draw()
    
def onpick2(event):
    ###Need to put outside variables as global because can't take them as input with mpl_connect
    global start
    global x_temp
    global y_temp
    global x1_pts
    global y1_pts
    global x2_pts
    global y2_pts
    m_x, m_y = event.x, event.y
    x, y = ax.transData.inverted().transform([m_x, m_y])
    if start==0:
        #First point : left line
        x1_pts.append(x)
        y1_pts.append(y)
        line1.set_xdata(x1_pts)
        line1.set_ydata(y1_pts)
        start+=1
    elif start==1:
        #Second point : right line
        x2_pts.append(x)
        y2_pts.append(y)
        line2.set_xdata(x2_pts)
        line2.set_ydata(y2_pts)
        start +=1
    else:
        #Check if next point is in the right direction  : above the line between 2 previous points
        p1 = np.asarray([x1_pts[-1], y1_pts[-1]])
        p2 = np.asarray([x2_pts[-1], y2_pts[-1]])
        temp = np.asarray([x, y])
        v_p = p2-p1
        v_d = p2-temp
        
        pos = np.cross(v_p, v_d)
        if pos>0 :
            print("Dessous")
        if pos<0 : 
            print("Dessus")
            #Store next single point
            x_temp.append(x)
            y_temp.append(y)
            pt_temp.set_xdata(x_temp)
            pt_temp.set_ydata(y_temp)
        if len(x_temp)==2:
            #When 2 points stored : determine orientation along quadrilatere so the track doesn't cross itself
            print("Update")
            temp1 = np.asarray([x_temp[0], y_temp[0]])
            temp2 = np.asarray([x_temp[1], y_temp[1]])
            v_temp = temp2-temp1
            print(v_p)
            print(v_temp)
            orientation = np.dot(v_p, v_temp)
            print(orientation)
            if orientation>0:
                x1_pts.append(x_temp[0])
                y1_pts.append(y_temp[0])
                x2_pts.append(x_temp[1])
                y2_pts.append(y_temp[1])
            elif orientation<0:
                x1_pts.append(x_temp[1])
                y1_pts.append(y_temp[1])
                x2_pts.append(x_temp[0])
                y2_pts.append(y_temp[0])
            line1.set_xdata(x1_pts)
            line1.set_ydata(y1_pts)
            line2.set_xdata(x2_pts)
            line2.set_ydata(y2_pts)
            x_temp = []
            y_temp = []
    fig.canvas.draw()
    
def onpick3(event):
    global x_temp
    global y_temp
    x_temp = x_temp[:-1]
    y_temp = y_temp[:-1]
    pt_temp.set_xdata(x_temp)
    pt_temp.set_ydata(y_temp)
    fig.canvas.draw()
    

fig.canvas.mpl_connect('button_press_event', onpick2)
fig.canvas.mpl_connect('key_press_event', onpick3)

plt.show()


def save_track(filename):
    np.save(filename, create_track(x1_pts, y1_pts, x2_pts, y2_pts))


def create_track(x1_pts, y1_pts, x2_pts, y2_pts):
    l1 = list(zip(x1_pts, y1_pts))
    l2 = list(zip(x2_pts, y2_pts))
    track = np.asarray([l1, l2])
    return track


def display(track):
    plt.figure(figsize=(9,9))
    plt.scatter(track[0][:,0], track[0][:,1])
    plt.plot(track[0][:,0], track[0][:,1])
    plt.scatter(track[1][:,0], track[1][:,1])
    plt.plot(track[1][:,0], track[1][:,1])
    plt.show()
    
def background(track, ax):
    track_ = np.concatenate((track, np.reshape([track[:,0]], (2, 1, 2))), axis=1)
    ax.scatter(track[0][:,0], track[0][:,1])
    ax.plot(track_[0][:,0], track_[0][:,1])
    ax.scatter(track[1][:,0], track[1][:,1])
    ax.plot(track_[1][:,0], track_[1][:,1])

def scale(track, ref=0.1):
    ###On veut avoir une largeur de piste à peu près constante pour que la voiture ait des inputs assez stables, on rescale donc pour que la largeur moyenne de la piste soit 0.1
    wide_av = np.mean(np.linalg.norm(track[0]-track[1], axis=1))
    print(wide_av)
    print('Min', np.min(np.linalg.norm(track[0]-track[1], axis=1)))
    print('Max', np.max(np.linalg.norm(track[0]-track[1], axis=1)))
    track_ = track*ref/wide_av
    wide_av_ = np.mean(np.linalg.norm(track_[0]-track_[1], axis=1))
    print(wide_av_)
    print('Min', np.min(np.linalg.norm(track_[0]-track_[1], axis=1)))
    print('Max', np.max(np.linalg.norm(track_[0]-track_[1], axis=1)))
    return track_

def center(track):
    bary = np.mean(np.concatenate((track[0], track[1]), axis=0), axis=0)
    track_ = track-bary
    return track_

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def place_car(car_position, car_speed, car_length, car_width, ax):
    pos_corner = [car_position[0]-car_width/2, car_position[1]-car_length/2]
    angle = np.angle(car_speed[0]+car_speed[1]*1j)
    angle_deg = np.angle(car_speed[0]+car_speed[1]*1j, deg=True)-90
    pos_corner = rotate(car_position, pos_corner, angle )
    car = patches.Rectangle(pos_corner, car_width, car_length, angle_deg)
    ax.add_patch(car)
    return car
    
    
def render(track, car_position=[0, 0], car_speed=[0.5, 0.5], car_length=0.025, car_width=0.015):
    plt.ion()
    fig, ax = plt.subplots()
    ax.axis('equal')
    background(track, ax)
    place_car(car_position, car_speed, car_length, car_width, ax)
    plt.show()
    
car_pos = np.linspace(-0.25, 0.25, 200)

def move(track, car_pos=np.linspace(-0.25, 0.25, 200), car_speed=[0.5, 0.5], car_length=0.025, car_width=0.015):
    plt.ion()
    fig, ax = plt.subplots()
    ax.axis('equal')
    background(track, ax)
    car=None
    t = time()
    for car_position in car_pos:
        if car is not None:
            car.remove()
        car = place_car([car_position, 0], car_speed, car_length, car_width, ax)
        plt.pause(0.001)
    plt.show()
    print(time()-t)
    
def reverse(track):
    l1, l2 = track
    x1 = l1[:,0]
    y1 = l1[:,1]
    x2 = l2[:,0]
    y2 = l2[:,1]
    x1_ = -x1
    x2_ = -x2
    track_ = create_track(x2_, y2, x1_, y1)
    return track_

def equalize(track, step=None):
    idx = list(range(1, track.shape[1])) + [0,]
    l1, l2 = track
    l1_dec = l1[idx]
    l2_dec = l2[idx]
    l1_len = np.linalg.norm(l1-l1_dec, axis=1)
    l2_len = np.linalg.norm(l2-l2_dec, axis=1)
    m_len = np.mean((l1_len, l2_len), axis=0)
    #print(m_len)
    print(np.mean(l1_len), np.min(l1_len), np.max(l1_len), np.std(l1_len))
    print(np.mean(l2_len), np.min(l2_len), np.max(l2_len), np.std(l2_len))
    if step is None:
        step = np.min(m_len)
    l1_ = np.copy(l1)
    l2_ = np.copy(l2)
    dec=0
    for i in range(track.shape[1]):
        n = int(m_len[i] / step)
        for j in range(1, n):
            l1_ = np.insert(l1_, i+1+dec, (1-j/n)*l1[i]+(j/n)*l1_dec[i], axis=0)
            l2_ = np.insert(l2_, i+1+dec, (1-j/n)*l2[i]+(j/n)*l2_dec[i], axis=0)
            dec+=1
    track_ = np.asarray([l1_.tolist(), l2_.tolist()])
    display(track_)
    return track_
        
    
    

