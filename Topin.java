/*==============================
 This is MOOCVis Project Code By GPLlab 
==============================*/
// Need G4P library
import g4p_controls.*;

// java libraries for read chinese(BIG5) and English(UTF8)
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.*;
import processing.video.*;

import javax.swing.BoundedRangeModel;
import javax.swing.JTextArea;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;
import javax.swing.JTextField;

import java.awt.geom.*; 
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;

/*============================================
lacation-
  search: shelly_per, shelly_eco, shelly_phi 
variable-
  Perception: 24(31:auto) Topics, 419 comments, Name:Cognitive Neuroscience
  Economics: 24(34:auto) Topics, 413 comments, Name:Introduce to Economics
  Philosophy: 21 Topics 420 comments
video end time-
  P---15:40(940.0)
  E---16:07(967.0)
  Ph--13:43(824.0)
excution set up-
  ./shelly_XXX/........
==============================================*/
public static final int COMMENT_NUM = 420;
public static final int TOPIC_NUM = 21;
public static final float VIDEO_LENGTH = 824.0; 
public static final String LOCAL_PATH = "./GUI_MOOCVis/shelly_phi/";
public static final String LOAD_TABLE = "comment.csv";
public static final String LOAD_KEYWORD = "keyword_v1.txt";
public static final String LOAD_Course_Name = "Philosophy";
/*======================================================                   ====================================================== */
/*====================================================== Declare variables ====================================================== */
/*======================================================       start       ====================================================== */
Table table;
public float xo,yo,xoT,yoT;
float zoom = 0.5, zoomT = 0.5;

// global data set
int n_point = COMMENT_NUM;
float[] point_t = new float[COMMENT_NUM];  // topic
int[]point_tp = new int[COMMENT_NUM] ;     // content type
int[] point_r = new int[COMMENT_NUM];      // related
int[] point_e = new int[COMMENT_NUM];      // emotion
int[] point_c = new int[COMMENT_NUM];      // content
Topic[] topics = new Topic[TOPIC_NUM];

// topic keywords
int box_wid = 200,box_leng = 200;
//String[] keyword = new String[TOPIC_NUM];

///Themeriver(bird) : new window
public int []themeriver = new int [20];
public int []themeriver2= new int [20];
public int[] stack1 = new int[20*9];
public int[] stack2 = new int[20*9];

public int[] emo_p = new int[20*9];
public int[] emo_n = new int[20*9];

public int Xlength=1900, Ylength=600; 

public int Xaxis = 200;
public int Yaxis = 450, Yaxis_th = 450;
public int t_dis =100, t_dis_th = 100;

// Event flag
boolean NoneClickTop = true;
public float low_opacity = 20; // default opacity value -> no highlight 
public float high_opacity = 200; // default opacity value -> no highlight 

//syncho video and topic_flow and theme_river ball(playing time())
float time_line_topic;
float time_line_theme;
float axis_weight = 5;

//GUI controller flag
public boolean vis_TopicFlow = true;
public boolean vis_ThemeRiver = false;
public int cate_mode = 1; // 1: related, 2: emotion
public boolean[] cate_content = new boolean[9];  // 1~8
public int com_filter = 0;
public int topic_y_mode = 0;  // y axis mode flag for topic flow 0~1
public int theme_y_mode = 0;  // y axis mode flag for theme river 0~1
int max_num = 60;
//GUIs

/* video */
public Movie theMov; 
public PImage play, stop;

public float video_wid = 300, video_leng = 145;
public float video_x =120, video_y =30;
public float press_size = 19;
public float barlength = 10;
public float barwidth = video_wid-2*press_size;
public float barstart = video_x + 2*press_size;
public float barend = barstart+barwidth;

public float play_press_x, play_press_y;
public float stop_press_x, stop_press_y;

public float radius = 10;
public float ball_x = 0;

public boolean isPlaying = true;
public boolean isStoping = false;
public boolean isBarPressing = false;

public float mx, my;

// comment 用
public int falsecount = TOPIC_NUM;
boolean displayIcon = true;
// read words file
List<String> lines ;
//List <Integer> showcom ;
List<String> keywords ;
List<String> usernames;

PFont myFont;
public PFont font;

/*======================================================                   ====================================================== */
/*====================================================== Declare variables ====================================================== */
/*======================================================        end        ====================================================== */

/*======================================================                     ====================================================== */
/*====================================================== Set up + Read files ====================================================== */
/*======================================================        start        ====================================================== */

public void setup(){
  //size(1366, 768, JAVA2D);
  size(1100,450,JAVA2D);
  frame.setLocation(0,0);
  
  // Place your setup code here
  xo = 0;
  yo = 0;
  xoT = 0;
  yoT = -77;
    
  // set Font...
  myFont = createFont("微軟正黑體",90);
  font = loadFont(LOCAL_PATH + "MS-Gothic-20.vlw");
  //textFont(font, 20);
  
  readcomment();
  readtopkey();
  textFont(myFont);
  //System.out.println(keywords);
    
  frameRate(4); // frame rate
  //Read file
  //table = loadTable(LOCAL_PATH + "comment.csv", "header"); 
  table = loadTable(LOCAL_PATH + LOAD_TABLE, "header"); 
  int i=0;
  for (TableRow row : table.rows()) {
    point_t[i] = row.getFloat("time");  
    point_tp[i] = row.getInt("topic");
    point_r[i] = row.getInt("related");
    point_e[i] = row.getInt("emotion");
    point_c[i] = row.getInt("content");
    i++;
  }
  
  //copy load modify(bird)
  /*
  String[] stuff1 = loadStrings(LOCAL_PATH + "related_content.txt");
  related = int(split(stuff1[0],','));
  String[] stuff2 = loadStrings(LOCAL_PATH + "unrelated_content.txt");
  unrelated = int(split(stuff2[0],','));
  String[] stuff3 = loadStrings(LOCAL_PATH + "emo_p_content.txt");
  emo_p = int(split(stuff3[0],','));
  String[] stuff4 = loadStrings(LOCAL_PATH + "emo_n_content.txt");
  emo_n = int(split(stuff4[0],','));
  */

  //Topics + content initialization 
  topic_set();  
  for(int q=0; q<9; q++){
    cate_content[q] = true;    
  }
  
  try {
    Thread.sleep(3000); //1000 milliseconds is one second.
  } catch(InterruptedException ex) {
      Thread.currentThread().interrupt();
  }
  
  // video set up
  video_setup();
  
  //Video, TF, TR 
  time_line_topic = theMov.duration()/60*t_dis;  //t_dis = 100
  time_line_theme = theMov.duration()/60*t_dis_th; // t_dis_th = 100
  
  GUI();
  GUIFontSet();
  
  /*float time=millis();
  float CCounter=0;
  while(millis() - time < 500){
    //println("tick");//if it is, do something
    //time = millis();//also update the stored time
    CCounter=millis() - time;
    print(CCounter+"\n");
  }
  //print(CCounter+"\n");
  // comment initialization*/
  comment_Init();
}

void GUI(){
  createGUI();
  customGUI();
}

void readcomment(){
  // For comment
  lines = new ArrayList<String>();
  int count=0;
  BufferedReader reader = null;
  try {
    reader = new BufferedReader(new InputStreamReader(new FileInputStream(LOCAL_PATH + "comment_v1.txt"), "UTF-8")); 
    String str = null;
    while ((str = reader.readLine()) != null) {
      //System.out.println(str);
      lines.add(str);
    }
  }catch (FileNotFoundException e) {
    e.printStackTrace();
  }catch (IOException e) {
    e.printStackTrace();
  }
  // For username
  usernames = new ArrayList<String>();
  count=0;
  BufferedReader reader2 = null;
  try {
    reader2 = new BufferedReader(new InputStreamReader(new FileInputStream(LOCAL_PATH + "username_v1.txt"), "UTF-8"));
    String str = null;
    while ((str = reader2.readLine()) != null) {
      //System.out.println(str);    
      usernames.add(str);
    }
  }catch (FileNotFoundException e) {
    e.printStackTrace();
  }catch (IOException e) {
    e.printStackTrace();
  }
}

void readtopkey(){
  // For keyword
  keywords = new ArrayList<String>();
  int count=0;
  BufferedReader reader = null;
  try {
    reader = new BufferedReader(new InputStreamReader(new FileInputStream(LOCAL_PATH + LOAD_KEYWORD), "UTF-8")); 
    String str = null;
    while ((str = reader.readLine()) != null) {
      //System.out.println(str);    
      keywords.add(str);
    }
  }catch (FileNotFoundException e) {
    e.printStackTrace();
  }catch (IOException e) {
    e.printStackTrace();
  }
}

public void video_setup(){
  theMov = new Movie(this, LOCAL_PATH + "345.mp4");
  play = loadImage(LOCAL_PATH + "play.jpg");
  stop = loadImage(LOCAL_PATH + "stop.jpg");
  theMov.play();  //plays the movie once
  //println(theMov.duration());
  //
  play_press_x = video_x;
  play_press_y = video_y+video_leng;
  stop_press_x = video_x+press_size;
  stop_press_y = video_y+video_leng;
}

public void movieEvent(Movie m) { 
  m.read(); 
}

void Controller(){
  // Category mode
  if(check_related.isSelected()==true){
    cate_mode=1;
    check_emotion.setSelected(false);
  }
  else if(check_related.isSelected()==false){
    cate_mode=2;
    check_emotion.setSelected(true);
  }
  // Cate content 1~8
  if(check_GeneralSpeaking.isSelected()==true)  cate_content[0]=true;
    else{ check_GeneralSpeaking.setSelected(false); cate_content[0]=false;}
  if(check_Note.isSelected()==true)  cate_content[1]=true;
    else{ check_Note.setSelected(false);  cate_content[1]=false; }
  if(check_Opinion.isSelected()==true)  cate_content[2]=true;
    else{ check_Opinion.setSelected(false);  cate_content[2]=false;}
  if(check_Question.isSelected()==true)  cate_content[3]=true;
    else{ check_Question.setSelected(false); cate_content[3]=false; }
  if(check_Complain.isSelected()==true)  cate_content[4]=true;
    else{ check_Complain.setSelected(false);  cate_content[4]=false;}
  if(check_Compliment.isSelected()==true)  cate_content[5]=true;
    else{ check_Compliment.setSelected(false);  cate_content[5]=false; }
  /*if(check_Agree.isSelected()==true)  cate_content[6]=true;
    else{ check_Agree.setSelected(false);  cate_content[6]=false; }
  if(check_Disagree.isSelected()==true)  cate_content[7]=true;
    else{ check_Disagree.setSelected(false);  cate_content[7]=false; }*/
  // vis method
  
  // comment filter
  if(vis_TopicFlow == true){ // topic flow
    /*slider_CommentFilter.setAlpha(250);
    slider_CommentFilter.setEnabled(true);
    com_filter=slider_CommentFilter.getValueI();*/
  }
  // dropList Topic Flow
  if(vis_TopicFlow==true){ // Topic flow visualization
    dropList_TopicFlow.setEnabled(true);
    dropList_TopicFlow.setAlpha(250);
    if(vis_ThemeRiver == false){
      dropList_ThemeRiver.setEnabled(false);
      dropList_ThemeRiver.setAlpha(50);
    }
    if(dropList_TopicFlow.getSelectedIndex()==0) //
      topic_y_mode=0;
    else if(dropList_TopicFlow.getSelectedIndex()==1) //
      topic_y_mode=1;
  }
  if (vis_ThemeRiver == true) {  // Theme river visualization
    dropList_ThemeRiver.setEnabled(true);
    dropList_ThemeRiver.setAlpha(250);
    if(dropList_ThemeRiver.getSelectedIndex()==0) //
      theme_y_mode=0;
    else if(dropList_ThemeRiver.getSelectedIndex()==1) //
      theme_y_mode=1;
  }
  if(falsecount==TOPIC_NUM){
    //text_comment.setText("");
  }  
}

/*======================================================                     ====================================================== */
/*====================================================== Set up + Read files ====================================================== */
/*======================================================         end         ====================================================== */

public void draw(){
  background(245);
  IconColor_draw();
  translate(xo,yo);
  scale(zoom);
  //translate (leftright,updown);
  //pushMatrix();
  Controller();
  // control show what visualization
  if(vis_TopicFlow == true){
    axis();  
    point_draw();
    line_draw();   
    topic_draw();

    topic_line_ball_draw();

    box_draw();
  }

  float abs_mou_x = (mouseX -xo)/zoom;
  float abs_mou_y = (mouseY -yo)/zoom;
  if(abs_mou_x > Xaxis && abs_mou_x<Xaxis+Xlength
     && abs_mou_y > Yaxis- axis_weight && abs_mou_y < Yaxis+ axis_weight){ //coordx
    float time = (abs_mou_x-Xaxis)/time_line_topic*theMov.duration();
    if(time<=VIDEO_LENGTH+1){
      strokeWeight(1);
      stroke(255,50);
      fill(255,50);
      rect(abs_mou_x-5,abs_mou_y-28,85,33);
      textFont(font,30);
      fill(0, 102, 153);   
      text((int)time/60+":"+(int)time%60,abs_mou_x,abs_mou_y);  
      textFont(myFont);
    }
  }
  //popMatrix();
}

void topic_line_ball_draw(){
    float start = Xaxis + radius/2; 
    float end = Xaxis+time_line_topic;
    float dis = end-start;
    float movie_legnth = theMov.duration();    
    float mt = theMov.time();
 
    if(isPlaying == false){
      ball_x = 0;
    }
    else {
      ball_x = (mt/movie_legnth)*dis;
    }
    fill(255);
    strokeWeight(1);
    ellipse(start+ball_x,Yaxis,radius,radius); 
    // 15:40 = 940s
    float endline_x = (VIDEO_LENGTH/movie_legnth)*dis;
    strokeWeight(5);
    line(start+endline_x,Yaxis+15,start+endline_x,Yaxis-15);
}

void theme_line_ball_draw(GWinApplet appc){
    float start = Xaxis + radius/2; 
    float end = Xaxis+time_line_theme;
    float dis = end-start;
    float movie_legnth = theMov.duration();    
    float mt = theMov.time();
     
    if(isPlaying == false){
      ball_x = 0;
    }
    else {
      ball_x = (mt/movie_legnth)*dis;
    }
    appc.fill(255);
    appc.strokeWeight(1);
    appc.ellipse(start+ball_x,Yaxis,radius,radius);
    // 15:40 = 940s
    float endline_x = (VIDEO_LENGTH/movie_legnth)*dis;
    appc.strokeWeight(5);
    appc.line(start+endline_x,Yaxis+15,start+endline_x,Yaxis-15);
}

void keyPressed() {  // 調整zoom in/out
   if (key == CODED) {
       if (keyCode == UP) {
       zoom = zoom+0.3; 
       } else 
       if (keyCode == DOWN) {
           zoom = zoom-0.3;
       } 
   }
   if(key == ' ') {
    zoom = 0.5;
    xo = 0;
    yo = 0;
   }
   if(key==ENTER){ // Icon or not
     if(displayIcon==true)
       displayIcon=false;
     else displayIcon=true;
   }   
}

void mouseWheel(MouseEvent event) {
  float rotatedCount = event.getCount();
  //println(rotatedCount);
  if(rotatedCount>0)
    zoom = zoom-0.1;
  else if(rotatedCount<0)
    zoom = zoom+0.1;
}

void mouseDragged(){
    xo = xo + (mouseX - pmouseX);
    yo = yo +(mouseY - pmouseY);
}

void mouseClicked(MouseEvent Meven) {
    
  falsecount = 0;
  float abs_mou_x = (mouseX -xo)/zoom;
  float abs_mou_y = (mouseY -yo)/zoom;
  float abs_top_x = 0.0; // top_left corner point
  float abs_top_y = 0.0; // 
  float top_width = 0.0;
  float top_height = 0.0;
          
  for(int i=0; i<TOPIC_NUM; i++){    // check all topic
        float rec_width = topics[i].get_number()*4;
        float x =  topics[i].get_coordx()/60*t_dis;
        float y =0;
        int rate = 1;
        //related 
        if(topic_y_mode == 0){
          y = (float) ((topics[i].get_related()-1.5)*Ylength)*0.6;
          if(y<0){ y-=rec_width/2; }
          else {y+=rec_width/2;}
        }
        else if(topic_y_mode == 1){ //emotion      
          y = (-1*topics[i].get_emotion())*Ylength/2;
          if(y<0){ y-=rec_width/2; }
          else{y+=rec_width/2;}
        }
    
      abs_top_x = Xaxis+x;  
      abs_top_y = Yaxis+y*rate;
      top_width = rec_width;
      top_height = rec_width/2;
    
      if(mouseButton == LEFT || mouseButton == RIGHT ) { // highlight function
          if(NoneClickTop == true){ // no top be clicked
                if(abs_mou_x<=abs_top_x+rec_width && abs_mou_x>=abs_top_x && 
                    abs_mou_y<=abs_top_y+top_height && abs_mou_y>=abs_top_y){ // click me                        
                    if(topics[i].DraworNot==true){
                          topics[i].DraworNot=true;
                    }
                }
                else { // click not me
                    topics[i].DraworNot=false;
                    falsecount=falsecount+1;
                }
          }
          else { // more than one of all topics is(are) clicked
                if(abs_mou_x<=abs_top_x+rec_width && abs_mou_x>=abs_top_x && 
                    abs_mou_y<=abs_top_y+top_height && abs_mou_y>=abs_top_y){ // click me                        
                    if(topics[i].DraworNot==true){ // i am clicked topic previousely
                          //print(topics[i].DraworNot);
                          topics[i].DraworNot=false; 
                          //falsecount=falsecount+1;
                    }
                    else // i am not the clicked topic previousely
                          topics[i].DraworNot=true;
                }
                else { // click not me
                    if(topics[i].DraworNot==true){
                          topics[i].DraworNot=false;
                          falsecount=falsecount+1;
                    }
                    else{
                        topics[i].DraworNot=false;
                        falsecount=falsecount+1;
                    }
                }
          }
      } // end highlight
      
          
      //bird spicy
      if(mouseButton == LEFT || mouseButton == RIGHT ){ //          
          if(abs_mou_x<=abs_top_x+rec_width && abs_mou_x>=abs_top_x && 
             abs_mou_y<=abs_top_y+top_height && abs_mou_y>=abs_top_y){ // click me                              
              if(topics[i].get_rightpress()==false)      
                  topics[i].isrightpressing();
              else
                  topics[i].notrightpressing();                       
          }
          else{
              topics[i].notrightpressing();
          }
      } // end box
  } // end all topic loop
        
  // bird hothot : check click bar (control video ball)
  if(abs_mou_x > Xaxis && abs_mou_x<Xaxis+Xlength
    && abs_mou_y > Yaxis- axis_weight && abs_mou_y < Yaxis+ axis_weight){ //coordx{
      //print("timeB: "+abs_mou_x+","+Xaxis+","+time_line_topic+","+theMov.duration()+" ,Xlengh(TF): "+Xlength+"\n");
      theMov.jump((abs_mou_x-Xaxis)/time_line_topic*theMov.duration());
      theMov.pause();  isStoping = true;
      //print("time: "+theMov.time()+" ,Xlengh(TF): "+Xlength+"\n");  
  }     
  
  // final check highlight 
  if(falsecount==TOPIC_NUM){ 
      //text_comment.setText("");
      for(int q=0; q<TOPIC_NUM; q++){
          topics[q].DraworNot=true;
      }
      NoneClickTop=true;
      //falsecount=0;
      comment_text();
   }
   else { NoneClickTop=false; comment_text(); } // all false的情況也包括
   falsecount=0;
}

void comment_Init(){
  ArrayList showcom = new ArrayList<Integer>();
  int number=0;
  for(int co=0; co<COMMENT_NUM; co++){
      if(point_c[co]!=9 && point_tp[co]>0){  // content type filter
          //print(point_tp[co]+", ");
          //if(point_tp[co]>0){
              number++;
              showcom.add(co);
              float time = point_t[co];
              int min = int(time/60);
              int sec = int(time%60);
          //}
      }
  }
  //print("Init:"+number+", ");
  Show(showcom);
}

void comment_text(){
  ArrayList showcom = new ArrayList<Integer>();
  
  for(int co=0; co<COMMENT_NUM; co++){
      if(cate_content[point_c[co]-1]==true && point_c[co]!=9 ){  // content type filter
          //print(point_tp[co]+", ");
          if(point_tp[co]>0)
              if( topics[point_tp[co]-1].get_number()>=com_filter && (NoneClickTop==true || topics[point_tp[co]-1].DraworNot==true) ){ // topic filter
                  showcom.add(co);
                  float time = point_t[co];
                  int min = int(time/60);
                  int sec = int(time%60);
              }
      }
  }
  Show(showcom);
}

public void Show(ArrayList<Integer> showComment){  
  String cmdd;//="";
  String[] cmdd2;// = new String[showComment.size];
  boolean checkappend=false;
  //text_comment.setText("");
  int m = 0;//int(point_t[showcom.get(d)]/60);
  int s = 0;//int(point_t[showcom.get(d)]%60);
  if(showComment.size()>0){
    if(showComment.size()<=300){
      //print("<=200:"+showComment.size()+", ");
      m = int(point_t[showComment.get(0)]/60);
      s = int(point_t[showComment.get(0)]%60);
      cmdd = m+":"+s+" "+usernames.get(showComment.get(0))+"-> "+lines.get(showComment.get(0))+"\n";
      //print(cmdd[0]+"\n");
      //text_comment.appendText(cmdd[0]);
      for(int c=1;c<showComment.size();c++)
        if(cmdd!=null){
          m = int(point_t[showComment.get(c)]/60);
          s = int(point_t[showComment.get(c)]%60);
          if(c==showComment.size()-1)
            cmdd = cmdd + m+":"+s+" "+usernames.get(showComment.get(c))+"-> "+lines.get(showComment.get(c));
          else
            cmdd = cmdd + m+":"+s+" "+usernames.get(showComment.get(c))+"-> "+lines.get(showComment.get(c))+"\n";
        }
      //print(cmdd);
      //if(cmdd!=null){
        //text_comment.clearStyles(0);
        text_comment.setText(cmdd);
        //text_comment.StyledString(cmdd);
      //}
      //else
        //print("Text is null");
      
      cmdd = null;
    }
    else{ // comment >200
      print(">200, ");
      /*cmdd = new String[200];
      m = int(point_t[showComment.get(0)]/60);
      s = int(point_t[showComment.get(0)]%60);
      cmdd[0] = m+":"+s+" "+usernames.get(showComment.get(0))+"-> "+lines.get(showComment.get(0))+"\n";
      //text_comment.setText(cmdd[0]);
      /*for(int c=1;c<200;c++){
        if(cmdd!=null){
          m = int(point_t[showComment.get(c)]/60);
          s = int(point_t[showComment.get(c)]%60);
          cmdd[c] = m+":"+s+" "+usernames.get(showComment.get(c))+"-> "+lines.get(showComment.get(c))+"\n";
          //checkappend=text_comment.appendText(cmdd[c]+"\n");
          //print(cmdd[c]+"\n");
        }
      }
      //print(cmdd[showComment.size()-1]);
      if(cmdd!=null){
        text_comment.setText(cmdd);
        //text_comment.StyledString(cmdd);
      }
      else
        print("Text is null");*/
      // then: other comment
      //cmdd = null;
      /*cmdd2 = new String[showComment.size()-200];
      m = int(point_t[showComment.get(200)]/60);
      s = int(point_t[showComment.get(200)]%60);
      cmdd2[0] = m+":"+s+" "+usernames.get(showComment.get(200))+"-> "+lines.get(showComment.get(200))+"\n";
      /*text_comment.appendText(cmdd2[0]);
      for(int c=1;c<(showComment.size()-200);c++)
        if(cmdd2!=null){
          m = int(point_t[showComment.get(c+200)]/60);
          s = int(point_t[showComment.get(c+200)]%60);
          cmdd2[c] = m+":"+s+" "+usernames.get(showComment.get(c+200))+"-> "+lines.get(showComment.get(c+200))+"\n";
          text_comment.appendText(cmdd2[c]);
        }
      */
    }
  } 
}

// themeRiver + Draw
void draw_curve(int n[],int a,GWinApplet appc){
  int i = 0;
  int time =0;
  //curveTightness(-5.0);
  appc.beginShape();
  appc.curveVertex(Xaxis+time, Yaxis_th+n[i]*3*a); // 頭
  for (i=0; i < 20; i ++ ){
     appc.curveVertex(Xaxis+time, Yaxis_th+n[i]*3*a); 
    time += t_dis_th;
  }
  //strokeWeight(0.5);
  appc.curveVertex(Xaxis+time, Yaxis_th+n[i-1]*3*a); // 尾
  appc.endShape();
}

// using "point_r" and "point_e" 
void themeriver_draw(GWinApplet appc){
  int count1=0;
  int count2=0;
  int[] color_stack1 =new int[9];
  int[] color_stack2 =new int[9];
  // 17個timestep
  int[][] sum_related = new int[9][20]; 
  int[][] sum_unrelated = new int[9][20];
  int[][] sum_pemotion = new int[9][20];
  int[][] sum_nemotion = new int[9][20];
  for(int qq=0; qq<9; qq++){
    for(int pp=0;pp<20;pp++){
      sum_related[qq][pp]=0; //print(sum_related[qq][pp],",");
      sum_unrelated[qq][pp]=0; //print(sum_unrelated[qq][pp],",");
      sum_pemotion[qq][pp]=0; //print(sum_related[qq][pp],",");
      sum_nemotion[qq][pp]=0; //print(sum_unrelated[qq][pp],",");
    }   
  }
  int temp_t=0;
  ////whicn y axis mode
    for(int p=0; p<COMMENT_NUM; p++){ //統計量
      //if(point_c[p]!=100) // valid content type
        if(point_tp[p] != 0){ // 有的沒有topic
          if(topics[point_tp[p]-1].number >= com_filter)
            if(topics[point_tp[p]-1].DraworNot == true){ // filter and highlight
              //if(cate_content[point_c[p]-1]==true){
                temp_t=(int)point_t[p]/60+1;
                if(point_r[p]==1){ // related points && content is selected
                  sum_related[point_c[p]-1][temp_t]++; 
                }
                else if(point_r[p]==2){ // unrelated points && content is selected 
                  sum_unrelated[point_c[p]-1][temp_t]++;
                }
                if(point_e[p]==1){ // positive emotion points && content is selected
                  sum_pemotion[point_c[p]-1][temp_t]++; 
                }
                else if(point_e[p]==-1){ // negative emotion points && content is selected
                  sum_nemotion[point_c[p]-1][temp_t]++;
                }
              //}
            }// end filter, highlight
        }
    }// 統計完了 
    /*print("sum_related: \n");
    for(int qq=0; qq<8; qq++){
      for(int pp=0;pp<17;pp++){
        print(sum_related[qq][pp],", ");
      }
      print("\n");
    }*/
    if(theme_y_mode==0){  // related 不應該在外層?      
      for(int i =0;i<9;i++) { //draw
        //if(cate_content[i]==true){ 
          for(int j=0;j<20;j++){
            themeriver[j] += sum_related[i][j];
            stack1[count1*20+j] = sum_related[i][j];
          }
          color_stack1[count1] = i;    
          count1++;
        //}
      }
      count1--;
      for(int i =0;i<9;i++){
        //if(cate_content[i]==true){ //draw
          for(int j=0;j<20;j++){
            themeriver2[j] += sum_unrelated[i][j];
            stack2[count2*20+j] = sum_unrelated[i][j];                
          }
          color_stack2[count2] = i;
          count2++;
        //}
      }
      count2--;
    }
    if(theme_y_mode==1){  // emotion  
      for(int i =0;i<9;i++) { //draw
        //if(cate_content[i]==true){ 
          for(int j=0;j<20;j++){
            themeriver[j] += sum_pemotion[i][j];
            stack1[count1*20+j] = sum_pemotion[i][j];
          }
          color_stack1[count1] = i;    
          count1++;
        //}
      }
      count1--;
      for(int i =0;i<9;i++){
        //if(cate_content[i]==true){ //draw
          for(int j=0;j<20;j++){
            themeriver2[j] += sum_nemotion[i][j];
            stack2[count2*20+j] = sum_nemotion[i][j];                
          }
          color_stack2[count2] = i;
          count2++;
        //}
      }
      count2--;
    }
    ///draw curve
    appc.stroke(0,50);
    appc.strokeWeight(1.5); 
    float po_opacity=200;
    
    for(;count1>=0;){    
    //print(count1,", ");
      if(color_stack1[count1] == 0 && cate_content[0]==true) appc.fill(#F5FA08);  //general : yellow
      else if(color_stack1[count1] == 1 && cate_content[1]==true) appc.fill(#FFB7DD/*#6C4A2E*/);  //note : coffee
      else if(color_stack1[count1] == 2 && cate_content[2]==true) appc.fill(#3BF2E4);  //opin : thin blue
      else if(color_stack1[count1] == 3 && cate_content[3]==true) appc.fill(#0634D3);  //ques : blue
      else if(color_stack1[count1] == 4 && cate_content[4]==true) appc.fill(#E01219);  //cpln : red
      else if(color_stack1[count1] == 5 && cate_content[5]==true) appc.fill(#5FFF1C);  //cpli: green
      /*else if(color_stack1[count1] == 6 && cate_content[6]==true) appc.fill(#FA8108);  //agree : orange
      else if(color_stack1[count1] == 7 && cate_content[7]==true) appc.fill(#AD06D3);  //disaree : purple*/
      else if(color_stack1[count1] == 8 && (cate_content[0]==true && cate_content[1]==true &&
                cate_content[2]==true && cate_content[3]==true && cate_content[4]==true &&
                cate_content[5]==true && cate_content[6]==true && cate_content[7]==true) )  
              appc.fill(200);
      else appc.fill(200);
      draw_curve(themeriver,-1,appc);
      for(int p=0;p<20;p++){
        themeriver[p]-=stack1[count1*20+p];
      }
      count1--;
    }
    for(;count2>=0;){
      if(color_stack2[count2] == 0 && cate_content[0]==true) appc.fill(#F5FA08);
      else if(color_stack2[count2] == 1 && cate_content[1]==true) appc.fill(#FFB7DD/*#6C4A2E*/);
      else if(color_stack2[count2] == 2 && cate_content[2]==true) appc.fill(#3BF2E4);
      else if(color_stack2[count2] == 3 && cate_content[3]==true) appc.fill(#0634D3);
      else if(color_stack2[count2] == 4 && cate_content[4]==true) appc.fill(#E01219); 
      else if(color_stack2[count2] == 5 && cate_content[5]==true) appc.fill(#5FFF1C);
      else if(color_stack2[count2] == 6 && cate_content[6]==true) appc.fill(#FA8108);
      else if(color_stack2[count2] == 7 && cate_content[7]==true) appc.fill(#AD06D3);
      else if(color_stack1[count2] == 8 && (cate_content[0]==true && cate_content[1]==true &&
                cate_content[2]==true && cate_content[3]==true && cate_content[4]==true &&
                cate_content[5]==true && cate_content[6]==true && cate_content[7]==true) )  
              appc.fill(200);
      else appc.fill(200);
      draw_curve(themeriver2,1,appc);
      for(int p=0;p<20;p++){
        themeriver2[p]-=stack2[count2*20+p];
      }
      count2--;
    }
}


void line_draw(){
  strokeWeight(3);
  //int j =1;
  int[] count =new int[TOPIC_NUM];
  for(int i=0;i<TOPIC_NUM;i++){
    count[i]=0;
  }
  
  for(int j=0;j<COMMENT_NUM;j++){
    if(point_tp[j]!=0){
      
      float line_opacity=200; // default opacity
      
      int temp=0;
      int wid = count[point_tp[j]-1];
      float num = topics[point_tp[j]-1].get_number();
      float co_x = point_t[j]/60*t_dis;
      
      float rect_y = 0;
      if(topic_y_mode == 0){ //related
        rect_y= (topics[point_tp[j]-1].get_related()-1.5)*Ylength*0.6;
        if(rect_y<0){temp = -5;}
        else{rect_y+=num*4/2;temp = 5;}    
      }
      else if(topic_y_mode == 1){ //emotion
        rect_y= (-1*topics[point_tp[j]-1].get_emotion())*Ylength/2;
        if(rect_y<0){ temp = -5; }
        else{rect_y+=num*4/2; temp = 5;}
      }
      
      float rect_x=topics[point_tp[j]-1].get_coordx()/60*t_dis;
      float a = wid*4;
      noFill();
      
      int SerNumTop = point_tp[j];
      if(topics[SerNumTop-1].number>=com_filter){ //comment filter
        if(topics[SerNumTop-1].DraworNot==true){
          line_opacity=high_opacity;
        }
        else line_opacity=low_opacity;

        if(point_c[j]==1 && cate_content[0]==true) stroke(#F5FA08, line_opacity);    //general : yellow
        else if(point_c[j]==2 && cate_content[1]==true)stroke(#FFB7DD/*#6C4A2E*/, line_opacity); //note : coffee
        else if(point_c[j]==3 && cate_content[2]==true)stroke(#3BF2E4, line_opacity); //opin : thin blue
        else if(point_c[j]==4 && cate_content[3]==true)stroke(#0634D3, line_opacity); //ques : blue
        else if(point_c[j]==5 && cate_content[4]==true)stroke(#E01219, line_opacity); //cpln : red
        else if(point_c[j]==6 && cate_content[5]==true)stroke(#5FFF1C, line_opacity); //cpli: green
        /*else if(point_c[j]==7 && cate_content[6]==true)stroke(#FA8108, line_opacity); //agree : orange
        else if(point_c[j]==8 && cate_content[7]==true)stroke(#AD06D3, line_opacity); //disaree : purple*/
        else{stroke(125, line_opacity);} // no: gray  
        
        strokeWeight(2);
        bezier(Xaxis+co_x,Yaxis+temp,Xaxis+co_x,Yaxis+rect_y,Xaxis+rect_x+a,Yaxis,Xaxis+rect_x+a,Yaxis+rect_y);
          if(wid==0){ 
            strokeWeight(1);  
            stroke(0, line_opacity); 
            bezier(Xaxis+co_x,Yaxis+temp,Xaxis+co_x,Yaxis+rect_y,Xaxis+rect_x+a,Yaxis,Xaxis+rect_x+a,Yaxis+rect_y);
          }           
          else if(wid == num-1){ 
            strokeWeight(1);  
            stroke(0, line_opacity);
            bezier(Xaxis+co_x,Yaxis+temp,Xaxis+co_x,Yaxis+rect_y,Xaxis+rect_x+a,Yaxis,Xaxis+rect_x+a,Yaxis+rect_y);
          }
        count[point_tp[j]-1]+=1;  
      }
    }
  }// end all comments

}


void point_draw(){
  noStroke();
    for(int j=0;j<COMMENT_NUM;j++){
        float time = point_t[j];
        float posi = time/60*t_dis;

        float po_opacity=200; // default opacity
        
        if(point_tp[j]!=0){
          int SerNumTop = point_tp[j];  //get serial number of topic
          if(topics[SerNumTop-1].number>=com_filter){ // comment filter
            if(topics[SerNumTop-1].DraworNot==true){
              po_opacity=high_opacity;
            }
            else po_opacity=low_opacity;
            
            if(point_c[j]==1) fill(#F5FA08, po_opacity);    //general : yellow
            else if(point_c[j]==2)fill(#FFB7DD/*#6C4A2E*/, po_opacity); //note : coffee
            else if(point_c[j]==3)fill(#3BF2E4, po_opacity); //opin : thin blue
            else if(point_c[j]==4)fill(#0634D3, po_opacity); //ques : blue
            else if(point_c[j]==5)fill(#E01219, po_opacity); //cpln : red
            else if(point_c[j]==6)fill(#5FFF1C, po_opacity); //cpli: green
            /*else if(point_c[j]==7)fill(#FA8108, po_opacity); //agree : orange
            else if(point_c[j]==8)fill(#AD06D3, po_opacity); //disaree : purple*/
            else{fill(125, po_opacity);} // no: gray             
            //oStroke();
            ellipse( Xaxis+posi,Yaxis,4,4);
          }
        }
    }
}

// 畫矩形 and topics
void topic_draw(){ 
  float rec_opacity=200;   
  for(int i=0;i<TOPIC_NUM;i++){
    if( topics[i].number >= com_filter ){
      if( topics[i].DraworNot == true ){
        rec_opacity = high_opacity;
      } else {
        rec_opacity = low_opacity;
      }
      
      float rec_width = topics[i].get_number()*4;
      float x =  topics[i].get_coordx()/60*t_dis;
      float y =0;
      int rate = 1;
     
      // topics 相對於x asix 高度
      if(topic_y_mode == 0){ //related 
        y = (topics[i].get_related()-1.5)*Ylength*0.6;
        if(y < 0){ 
          y -= rec_width/2;
        } else{
          y += rec_width/2;
        }
      } else if(topic_y_mode == 1){ //emotion      
        y = (-1*topics[i].get_emotion())*Ylength/2;
        if(y<0){ y-=rec_width/2; }
        else{y+=rec_width/2;}
      }
      
      stroke(0, rec_opacity);
      strokeWeight(3);
      if(topic_y_mode==0){ // color is for related 
        float e =topics[i].get_emotion();
        if( e<=1 && e>0.5) fill(#E7E7E7,rec_opacity);
        else if(e<=0.5 && e>0) fill(#9F9F9F,rec_opacity);
        else if(e<=0 && e>(-0.5)) fill(#3F3F3F,rec_opacity);
        else if(e<=(-0.5) && e>= -1) fill(#0F0F0F,rec_opacity);
      }
      else if(topic_y_mode==1){ // color is for emotion
        float r =topics[i].get_related();
        if( r<=2 && r>1.75) fill(#0F0F0F, rec_opacity);
        else if(r<=1.75 && r>1.5) fill(#3F3F3F, rec_opacity);
        else if(r<=1.5 && r>(-1.25)) fill(#9F9F9F, rec_opacity);
        else if(r<=(-1.25) && r>= 1) fill(#E7E7E7, rec_opacity);       
      }

      rect(Xaxis+x,Yaxis+y*rate,rec_width,rec_width/2,10);
      topics[i].up_y(y);
    }
  }
}

void box_draw(){
    float box_x=0.0;  float box_y=0.0;
    int textinterval; int textsize;
    for(int i=0;i<TOPIC_NUM;i++){
        /*box_x =topics[i].get_coordx()/60*t_dis;
        box_y =topics[i].get_y()-box_leng;*/
        textinterval =30;
        textsize = 25;
        if(topics[i].get_rightpress()==true){ //be pressed
          String[] list = split(keywords.get(i), " ");
          box_x =topics[i].get_coordx()/60*t_dis;
          box_leng=list.length*textinterval+textsize/2;
          box_wid=textinterval*6;
 
          if(topic_y_mode==0){ // related
            //print(topics[i].get_related()+", ");
            if(topics[i].get_related()<=1.5)  // related
              box_y =topics[i].get_y()-(list.length*textinterval+textsize/2);
            else  // <0
              box_y =topics[i].get_y()+(topics[i].get_number()*2.2);              
          } 
          else if(topic_y_mode==1){ // emotion
            //print(topics[i].get_emotion()+", ");
            if(topics[i].get_emotion()>0.0)  // >=0
              box_y =topics[i].get_y()-(list.length*textinterval+textsize/2);
            else  // <0
              box_y =topics[i].get_y()+(topics[i].get_number()*2.2);
          }
          //draw bg rectangle
          fill(255,150);
          strokeWeight(0.5);
          rect(Xaxis+box_x,Yaxis+box_y,box_wid,box_leng);
          // text  
            //String[] list = split(keywords.get(i), " ");
            for(int j=0;j<list.length;j++)  {
                //print(list[j]+", ");
                fill(0);
                textSize(textsize);
                text(list[j],Xaxis+box_x+textsize/2,Yaxis+box_y+j*textinterval+textsize/0.8);
            }
        }
    }
}
// Icon 圖示提示
void IconColor_draw(){
  int opac;
  textSize(11);
  if(displayIcon==true)
    opac=150;
  else opac=50;
  // comment content type
  // 01:general...
  fill(#F5FA08,opac); strokeWeight(0); rect(945,5,12,12); 
  fill(0,57,82,opac); text("General",960,14);
  fill(0,57,82,opac); text("Conversation",960,27);
  // 02:general...
  fill(#FF8BDD/*#6C4A2E*/,opac); strokeWeight(0); rect(1019,5,12,12); 
  fill(0,57,82,opac); text("Note",1034,14);
  // 03:opinion...
  fill(#3BF2E4,opac); strokeWeight(0); rect(945,30,12,12); 
  fill(0,57,82,opac); text("Opinion",960,41);
  // 04:question...
  fill(#0634D3,opac); strokeWeight(0); rect(1019,30,12,12); 
  fill(0,57,82,opac); text("Question",1034,41);
  // 05:complain...
  fill(#E01219,opac); strokeWeight(0); rect(945,50,12,12); 
  fill(0,57,82,opac); text("Complain",960,61);
  // 06:compliment...
  fill(#5FFF1C,opac); strokeWeight(0); rect(1019,50,12,12); 
  fill(0,57,82,opac); text("Compliment",1034,61);
  // 07:agree...
  /*fill(#FA8108,opac); strokeWeight(0); rect(945,70,12,12); 
  fill(0,57,82,opac); text("Agree",960,80);
  // 08:disagree...
  fill(#AD06D3,opac); strokeWeight(0); rect(1019,70,12,12); 
  fill(0,57,82,opac); text("Disagree",1034,80);*/
  // topic size = # of comment
  // # of comment = 35
  textSize(12);
  fill(245,opac); strokeWeight(0); rect(1100-150*zoom,450-85*zoom,140*zoom,70*zoom,10);
  fill(0,57,82,opac); text("35",1100-150*zoom-20,450-45*zoom);
  // # of comment = 25
  fill(245,opac); strokeWeight(0); rect(1100-150*zoom,450-140*zoom,100*zoom,50*zoom,10);
  fill(0,57,82,opac); text("25",1100-150*zoom-20,450-107*zoom);
  // # of comment = 15
  fill(245,opac); strokeWeight(0); rect(1100-150*zoom,450-174*zoom,60*zoom,30*zoom,10);
  fill(0,57,82,opac); text("15",1100-150*zoom-20,450-148*zoom);
  // # of comment = 5
  fill(245,opac); strokeWeight(0); rect(1100-150*zoom,450-187*zoom,20*zoom,10*zoom,10);
  fill(0,57,82,opac); text("5",1100-150*zoom-17,450-176*zoom);
  
  // topic color for emo/rel
  textSize(12);
  fill(0,opac); strokeWeight(0); line(593,10,683,10);
  fill(#E7E7E7,opac); strokeWeight(0); rect(600,10,17,9);  
  if(topic_y_mode==0){
    fill(0,57,82,opac); text("positive",544,15);
    fill(0,57,82,opac); text("emotion",544,28); }
  else {
    fill(0,57,82,opac); text("related",545,20); }
    fill(#9F9F9F,opac); strokeWeight(0); rect(620,10,17,9);
    fill(#3F3F3F,opac); strokeWeight(0); rect(640,10,17,9);
    fill(#0F0F0F,opac); strokeWeight(0); rect(660,10,17,9);  
  if(topic_y_mode==0){
    fill(0,57,82,opac); text("negative",690,15);
    fill(0,57,82,opac); text("emotion",690,28);}
  else {
    fill(0,57,82,opac); text("unrelated",690,20);
    //fill(0,57,82,opac); text("relevance",690,28);
  }  
}

void IconColor_drawT(GWinApplet appc){
  int opac;
  appc.textSize(11);
  if(displayIcon==true)
    opac=150;
  else opac=50;
  // comment content type
  // 01:general...
  appc.fill(#F5FA08,opac); appc.strokeWeight(0); appc.rect(1165,25,12,12); 
  appc.fill(0,57,82,opac); appc.text("General",1180,34);
  appc.fill(0,57,82,opac); appc.text("Conversation",1180,47);
  // 02:general...
  appc.fill(#FF8BDD/*#6C4A2E*/,opac); appc.strokeWeight(0); appc.rect(1249,25,12,12); 
  appc.fill(0,57,82,opac); appc.text("Note",1264,34);
  // 03:opinion...
  appc.fill(#3BF2E4,opac); appc.strokeWeight(0); appc.rect(1165,50,12,12); 
  appc.fill(0,57,82,opac); appc.text("Opinion",1180,61);
  // 04:question...
  appc.fill(#0634D3,opac); appc.strokeWeight(0); appc.rect(1249,50,12,12); 
  appc.fill(0,57,82,opac); appc.text("Question",1264,61);
  // 05:complain...
  appc.fill(#E01219,opac); appc.strokeWeight(0); appc.rect(1165,70,12,12); 
  appc.fill(0,57,82,opac); appc.text("Complain",1180,81);
  // 06:compliment...
  appc.fill(#5FFF1C,opac); appc.strokeWeight(0); appc.rect(1249,70,12,12); 
  appc.fill(0,57,82,opac); appc.text("Compliment",1264,81);
  // 07:agree...
  /*appc.fill(#FA8108,opac); appc.strokeWeight(0); appc.rect(1165,90,12,12); 
  appc.fill(0,57,82,opac); appc.text("Agree",1180,100);
  // 08:disagree...
  appc.fill(#AD06D3,opac); appc.strokeWeight(0); appc.rect(1249,90,12,12); 
  appc.fill(0,57,82,opac); appc.text("Disagree",1264,100);*/
}

void axisT(GWinApplet appc){
    appc.strokeWeight(5);
    appc.stroke(0);
    appc.line(Xaxis-10,Yaxis,Xaxis+Xlength,Yaxis);
    appc.line(Xaxis-10,Yaxis-Ylength/2.5,Xaxis-10,Yaxis+Ylength/2.5);
    // y-axis triangle 
    int upx = Xaxis-10; int upy = Yaxis-(int)(Ylength/2.5);
    int downx = Xaxis-10; int downy = Yaxis+(int)(Ylength/2.5);
    appc.triangle(upx,upy-6,upx-3,upy,upx+4,upy);
    appc.triangle(downx,downy+6,downx-3,downy,downx+4,downy);
    // x-axis triangle 
    int rightx = Xaxis+Xlength; int righty = Yaxis;
    appc.triangle(rightx+6,righty,rightx,righty-3,rightx,righty+4);
    appc.textSize(32);
    appc.fill(0, 102, 153);
    appc.text("time", 220+Xlength, Yaxis+10);
   
    if(vis_ThemeRiver==true){
        appc.textSize(30);
        appc.text("Comment", 28, Yaxis-Ylength/3-20);
        appc.text("counts", 42, Yaxis-Ylength/3+10);
        appc.text("Comment", 28, Yaxis+10+Ylength/3+10);
        appc.text("Counts", 42, Yaxis+10+Ylength/3+40);
      if(theme_y_mode==0){ //rel
        appc.text("Related", 975, Yaxis-Ylength/3-30); 
        appc.text("unrelated", 975, Yaxis+Ylength/3+30);
        //appc.text("Relevance", 975, Yaxis+Ylength/3+60);                
      }
      else if(theme_y_mode==1){
        appc.text("Positive", 975, Yaxis-Ylength/3-30);
        appc.text("Emotion", 977, Yaxis-Ylength/3);
        appc.text("Negative", 975, Yaxis+Ylength/3+30);
        appc.text("emotion", 977, Yaxis+Ylength/3+60);      
      }
    }
    appc.fill(0);
    appc.line(Xaxis-15,Yaxis-15,Xaxis-1,Yaxis-15); // 5
    appc.text("5",Xaxis-55,Yaxis-5);
    appc.line(Xaxis-15,Yaxis-45,Xaxis-1,Yaxis-45); // 15
    appc.text("15",Xaxis-55,Yaxis-35);
    appc.line(Xaxis-15,Yaxis-75,Xaxis-1,Yaxis-75); // 25
    appc.text("25",Xaxis-55,Yaxis-65);
    appc.line(Xaxis-15,Yaxis-105,Xaxis-1,Yaxis-105); // 35
    appc.text("35",Xaxis-55,Yaxis-95);
    
    appc.line(Xaxis-15,Yaxis+15,Xaxis-1,Yaxis+15); // 5
    appc.text("5",Xaxis-55,Yaxis+25);
    appc.line(Xaxis-15,Yaxis+45,Xaxis-1,Yaxis+45); // 15
    appc.text("15",Xaxis-55,Yaxis+55);
    appc.line(Xaxis-15,Yaxis+75,Xaxis-1,Yaxis+75); // 25
    appc.text("25",Xaxis-55,Yaxis+85);
    appc.line(Xaxis-15,Yaxis+105,Xaxis-1,Yaxis+105); // 35
    appc.text("35",Xaxis-55,Yaxis+115);
}
void axis(){
    strokeWeight(5);
    stroke(0);
    line(Xaxis-10,Yaxis,Xaxis+Xlength,Yaxis); // x-axis
    line(Xaxis-10,Yaxis-Ylength/2,Xaxis-10,Yaxis+Ylength/2); // y-axis
    // y-axis triangle 
    int upx = Xaxis-10; int upy = Yaxis-Ylength/2;
    int downx = Xaxis-10; int downy = Yaxis+Ylength/2;
    triangle(upx,upy-6,upx-3,upy,upx+4,upy);
    triangle(downx,downy+6,downx-3,downy,downx+4,downy);
    // x-axis triangle
    int rightx = Xaxis+Xlength; int righty = Yaxis;
    triangle(rightx+6,righty,rightx,righty-3,rightx,righty+4);
    textSize(32);
    fill(0, 102, 153);
    text("time", 220+Xlength, Yaxis+10);
    
    if(vis_TopicFlow==true){
      if(topic_y_mode==0){
        //text("Relevance", 30, Yaxis+10);
        textSize(30);
        //text("Positive", 75, Yaxis-Ylength/3-30);
        text("related", 75, Yaxis-Ylength/3-30);
        text("unrelated", 45, Yaxis+10+Ylength/3+20);
        //text("Relevance", 40, Yaxis+10+Ylength/3+50);
      }
      else if(topic_y_mode==1){
        //text("Emotion", 58, Yaxis+10);
        textSize(30);
        text("Positive", 70, Yaxis-Ylength/3-35);
        text("Negative", 50, Yaxis+10+Ylength/3+20);
        text("Emotion", 60, Yaxis-Ylength/3);
        text("Emotion", 60, Yaxis+10+Ylength/3+55);
      }
    }
}

void topic_set(){
  //create topics
    for(int j=0;j<TOPIC_NUM;j++){
        topics[j] = new Topic();
    }
  // number of comment
    for(int j=0;j<COMMENT_NUM;j++){
      if(point_tp[j]!=0){ 
         topics[point_tp[j]-1].up_number();      // save number
         //if(point_tp[j]-1==19) print(j+", ");
         topics[point_tp[j]-1].up_ct(point_c[j]); //save contemt         
         topics[point_tp[j]-1].up_re(point_r[j]); //sum relat
         topics[point_tp[j]-1].up_em(point_e[j]); //sum emo
         topics[point_tp[j]-1].up_sum(point_t[j]); //sum time

         topics[point_tp[j]-1].add_comment(j); //record comment id
       }
    }
  // coordX : medium number's time = set_number/2  
}

class Topic{

  boolean DraworNot = true;  // default
  boolean box_press = false; // combine with DraworNot
  
  IntList comment_index= new IntList();
  
  int count =0;
  float y; //coordinate
  float number =0; // numbers of comment
  float related, emotion;
  float sum=0;
  int[] ct = new int[9];
   //ct1:general conversation ; ct2:note
   //ct3:opinion ; ct4: question
   //ct5:complain;compliment; ct7: agree; ct8: disagree
  Topic(){
    for(int i=0;i<9;i++) 
      ct[i]=0;
  }

  void up_number(){
    number+=1;
  }

  void up_ct(int a){
    //if(a!=100)
      ct[a-1]+=1;
  } 

  void up_re(int r){
    related+=r;
  }
  void up_em(int e){emotion+=e;}
  void up_sum(float time){sum+=time;}
  void up_count(){count+=1;}
  void up_y(float a){y = a;}

  void add_comment(int index){ comment_index.append(index); /*print(comment_index.get(index));*/ }
  
  float get_coordx(){return sum/number-number/2;}
  float get_number(){return number;}
  float get_related(){return related/number;}
  float get_emotion(){return emotion/number;}
  float get_ct(int c){return ct[c-1];}  
  int   get_count(){return count;}
  float get_y(){return y;}

  int  get_comment(int a){ return comment_index.get(a);}
  
  void isrightpressing(){ box_press =true; }
  void notrightpressing(){ box_press =false; } 
  boolean get_rightpress(){return box_press; }
}

void GUIFontSet(){ // 更改GUI_text_font!!!
  // win_video
  label_lecture.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,16));
  // win_comment
  text_comment.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,15));  
  // win_controller
  label_TR.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,13));
  label_TF.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,13));
  label_Category.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,13));
  label_CommentType.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,13));
  label_comment_count.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,13));
  label_file_IO.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,13));
  label_Visual_Mode.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,13));
  check_related.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,12));
  check_emotion.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,12));
  check_GeneralSpeaking.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,12));
  check_Note.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,12));
  check_Opinion.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,12));
  check_Question.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,12));
  check_Complain.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,12));
  check_Compliment.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,12));
  /*check_Agree.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,12));
  check_Disagree.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,12));*/
  button_All.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,12));
  button_Clear.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,12));
  dropList_TopicFlow.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,12));
  dropList_ThemeRiver.setFont(g4p_controls.FontManager.getFont("微軟正黑體",1,12));
  
}

public GWindow window_theme;
public void createWindows(){
  //int col;
  //window = new GWindow[3];
  window_theme = new GWindow(this, "Theme River", -6,0/*0, 480*/, 1336, 292, false, JAVA2D);
  window_theme.addDrawHandler(this, "TR_draw");
  window_theme.addMouseHandler(this, "TR_mouse");
  //window_theme.close();
  window_theme.setActionOnClose(G4P.CLOSE_WINDOW);  
}
void TR_draw(GWinApplet appc, GWinData data) { // window_theme
  appc.background(245);
  IconColor_drawT(appc);
  appc.translate(xoT,yoT);
  appc.scale(zoomT);
  

  themeriver_draw(appc);  
  axisT(appc);
  theme_line_ball_draw(appc);
  
  
  float abs_mou_x = (appc.mouseX -xoT)/zoomT;
  float abs_mou_y = (appc.mouseY -yoT)/zoomT;
  if(abs_mou_x > Xaxis && abs_mou_x<Xaxis+Xlength
     && abs_mou_y > Yaxis- axis_weight && abs_mou_y < Yaxis+ axis_weight){ //coordx
    float time = (abs_mou_x-Xaxis)/time_line_theme*theMov.duration();
    appc.fill(255);
    appc.strokeWeight(1);
    appc.rect(abs_mou_x-5,abs_mou_y-28,85,33);
    appc.textFont(font, 30);
    appc.fill(0, 102, 153);
    appc.text((int)time/60+":"+(int)time%60,abs_mou_x,abs_mou_y);  
    appc.textFont(myFont);
  }
}  
void TR_mouse(GWinApplet appc, GWinData data, MouseEvent mevent) { // window_theme
  if(mevent.getAction()==MouseEvent.WHEEL){
    float rotatedCount = mevent.getCount();
    if(rotatedCount>0)
      zoom = zoom-0.1;
    else if(rotatedCount<0)
      zoom = zoom+0.1;
  }
  if(mevent.getAction()==MouseEvent.DRAG){
    xoT = xoT + (appc.mouseX - appc.pmouseX);
    yoT = yoT +(appc.mouseY - appc.pmouseY);
  }
  float abs_mou_x = (appc.mouseX -xoT)/zoomT;
  float abs_mou_y = (appc.mouseY -yoT)/zoomT;
  ////// bird hothot : press themeriver bar 
 if(mevent.getAction()==MouseEvent.CLICK){ 
    if(abs_mou_x > Xaxis && abs_mou_x<Xaxis+Xlength
       && abs_mou_y > Yaxis- axis_weight && abs_mou_y < Yaxis+ axis_weight){ //coord
       if(vis_ThemeRiver==true){
         //print("timeB: "+theMov.time()+" ,Xlengh(TR): "+Xlength+"\n");
         theMov.jump((abs_mou_x-Xaxis)/time_line_theme*theMov.duration()); 
         theMov.pause();  isStoping = true;
         //print("time: "+theMov.time()+" ,Xlengh(TR): "+Xlength+"\n");  
       }
    }
    else {  // content type filter
      color c = appc.get((int)appc.mouseX, (int)appc.mouseY);
      //appc.print(hex(c));      
      if(c==#F5FA08){  //general : yellow
        for(int cc=0; cc<8; cc++) cate_content[cc]=false;
        cate_content[0]=true;
        check_GeneralSpeaking.setSelected(true); check_Note.setSelected(false);
        check_Opinion.setSelected(false); check_Question.setSelected(false); 
        check_Complain.setSelected(false); check_Compliment.setSelected(false);
        /*check_Agree.setSelected(false); check_Disagree.setSelected(false);*/
      }
      else if(c==#FFB7DD){  //note : coffee
        for(int cc=0; cc<8; cc++) cate_content[cc]=false;
        cate_content[1]=true;
        check_GeneralSpeaking.setSelected(false); check_Note.setSelected(true);
        check_Opinion.setSelected(false); check_Question.setSelected(false); 
        check_Complain.setSelected(false); check_Compliment.setSelected(false);
        /*check_Agree.setSelected(false); check_Disagree.setSelected(false);*/
      }
      else if(c==#3BF2E4){  //opin : thin blue
        for(int cc=0; cc<8; cc++) cate_content[cc]=false;
        cate_content[2]=true;
        check_GeneralSpeaking.setSelected(false); check_Note.setSelected(false);
        check_Opinion.setSelected(true); check_Question.setSelected(false); 
        check_Complain.setSelected(false); check_Compliment.setSelected(false);
        /*check_Agree.setSelected(false); check_Disagree.setSelected(false);*/
      }
      else if(c==#0634D3){  //ques : blue
        for(int cc=0; cc<8; cc++) cate_content[cc]=false;
        cate_content[3]=true;
        check_GeneralSpeaking.setSelected(false); check_Note.setSelected(false);
        check_Opinion.setSelected(false); check_Question.setSelected(true); 
        check_Complain.setSelected(false); check_Compliment.setSelected(false);
        /*check_Agree.setSelected(false); check_Disagree.setSelected(false);*/
      }
      else if(c==#E01219){  //cpln : red
        for(int cc=0; cc<8; cc++) cate_content[cc]=false;
        cate_content[4]=true;
        check_GeneralSpeaking.setSelected(false); check_Note.setSelected(false);
        check_Opinion.setSelected(false); check_Question.setSelected(false); 
        check_Complain.setSelected(true); check_Compliment.setSelected(false);
        /*check_Agree.setSelected(false); check_Disagree.setSelected(false);*/
      }
      else if(c==#5FFF1C){  //cpli: green
        for(int cc=0; cc<8; cc++) cate_content[cc]=false;
        cate_content[5]=true;
        check_GeneralSpeaking.setSelected(false); check_Note.setSelected(false);
        check_Opinion.setSelected(false); check_Question.setSelected(false); 
        check_Complain.setSelected(false); check_Compliment.setSelected(true);
        /*check_Agree.setSelected(false); check_Disagree.setSelected(false);*/
      }
      /*else if(c==#FA8108){  //agree : orange
        for(int cc=0; cc<8; cc++) cate_content[cc]=false;
        cate_content[6]=true;
        check_GeneralSpeaking.setSelected(false); check_Note.setSelected(false);
        check_Opinion.setSelected(false); check_Question.setSelected(false); 
        check_Complain.setSelected(false); check_Compliment.setSelected(false);
        check_Agree.setSelected(true); check_Disagree.setSelected(false);
      }
      else if(c==#AD06D3){  //disaree : purple
        for(int cc=0; cc<8; cc++) cate_content[cc]=false;
        cate_content[7]=true;
        check_GeneralSpeaking.setSelected(false); check_Note.setSelected(false);
        check_Opinion.setSelected(false); check_Question.setSelected(false); 
        check_Complain.setSelected(false); check_Compliment.setSelected(false);
        check_Agree.setSelected(false); check_Disagree.setSelected(true);
      }*/
      else{  // all
        for(int cc=0; cc<8; cc++) cate_content[cc]=true;

        check_GeneralSpeaking.setSelected(true); check_Note.setSelected(true);
        check_Opinion.setSelected(true); check_Question.setSelected(true); 
        check_Complain.setSelected(true); check_Compliment.setSelected(true);
        /*check_Agree.setSelected(true); check_Disagree.setSelected(true);*/
      }
    }
    comment_text();
 }
}

// Use this method to add additional statements
// to customise the GUI controls
public void customGUI(){
  
}