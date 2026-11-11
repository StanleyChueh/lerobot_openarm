import sys; sys.path[:0]=['/home/stanley/Stanley_ws/sfa_hf_export/battery/bsfa','/home/stanley/Stanley_ws/sfa_hf_export/render_v4','/home/stanley/Stanley_ws/lerobot_openarm/research/sim_free_augmentation','/home/stanley/Stanley_ws/sfa_hf_export/target']
import numpy as np, cv2, bdata, mesh, srecon
from sfa.robot import Robot, Camera
rb=Robot(); G=mesh.Geometry(rb, device="cuda:0"); SIDE=np.array(G.side+["X"])
cams={"inherited":Camera(),"fitted":srecon.calibrated_camera()}
rows=[]
for E,js in ((1,(0,150,330)),(5,(0,140,420))):
    ep=bdata.load(E,cams=(bdata.CAMS[0],)); F=ep.frames[bdata.CAMS[0]]
    for name,c in cams.items():
        tiles=[]
        for j in js:
            sp=Robot.split(ep.state[j:j+1]); Ms=G.placements(sp["R"][0],sp["L"][0],sp["gR"][0],sp["gL"][0]); _,lab,_=mesh.splat(G.world(Ms),G.gid,c)
            sd=SIDE[lab.cpu().numpy()]; im=F[j].copy()
            for s,col in (("R",(255,60,60)),("L",(60,160,255))):
                cs,_=cv2.findContours((sd==s).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cs,-1,col,2)
            cv2.putText(im,f"ep{E} t={j} {name}",(5,25),0,0.8,(255,255,0),2); tiles.append(im)
        rows.append(np.concatenate(tiles,1))
cv2.imwrite(sys.argv[1],cv2.cvtColor(cv2.resize(np.concatenate(rows,0),None,fx=0.5,fy=0.5),cv2.COLOR_RGB2BGR))
