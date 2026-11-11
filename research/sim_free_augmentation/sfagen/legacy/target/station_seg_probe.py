import sys; sys.path[:0]=['/home/stanley/Stanley_ws/sfa_hf_export/battery/bsfa','/home/stanley/Stanley_ws/lerobot_openarm/research/sim_free_augmentation']
import numpy as np, cv2, brecon
E=int(sys.argv[1]); out=sys.argv[2]
rec=brecon.BRecon(E, device="cuda:0")
tg=rec.ev["L"]["closes"][0]
tiles=[]
for j in (0, tg-20, tg+20, tg+60):
    f=rec.F[j]; r,b=rec.silhouettes(j)
    fg=np.abs(f.astype(int)-rec.plate.astype(int)).max(2)>30
    fg=cv2.morphologyEx(fg.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))>0
    fg&=~(cv2.dilate(r.astype(np.uint8),np.ones((5,5),np.uint8))>0)&~b
    T=rec.tcp("L", tg if j<tg else j+rec.lag); uv,_=rec.cam.project(T[:3,3][None]); u,v=uv[0]
    ov=f.copy(); ov[fg]=(ov[fg]*0.4+np.array([255,0,255])*0.6).astype(np.uint8)
    ed=cv2.Canny(r.astype(np.uint8)*255,50,150)>0; ov[ed]=(0,255,0)
    cv2.circle(ov,(int(u),int(v)),6,(255,255,0),-1); cv2.putText(ov,f"t={j}",(5,25),0,0.8,(255,255,0),2)
    tiles.append(ov)
tiles.append(rec.plate)
cv2.imwrite(out,cv2.cvtColor(cv2.resize(np.concatenate(tiles,1),None,fx=0.5,fy=0.5),cv2.COLOR_RGB2BGR))
