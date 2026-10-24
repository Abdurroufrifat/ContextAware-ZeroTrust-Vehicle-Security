"""Rigid-body velocity translation, using angular velocity and displacement in one frame."""
def velocity_at_offset(velocity, angular, offset):
    vx, vy, vz = velocity
    wx, wy, wz = angular
    rx, ry, rz = offset
    return (vx + wy*rz - wz*ry, vy + wz*rx - wx*rz, vz + wx*ry - wy*rx)
