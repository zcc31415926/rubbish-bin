import java.util.*;


public class ScannerSampleInput
{
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("please input an integer:");
        int a = sc.nextInt();
        System.out.println(a);
        System.out.println("please input a float number:");
        float b = sc.nextFloat();
        System.out.println(b);
        System.out.println("please input a string:");
        String fuck = sc.nextLine();
        String c = sc.nextLine();
        System.out.println(c);
    }
}


//nextByte(),nextDouble(),nextFloat(),nextInt(),nextLine(),nextLong(),nextShort()
//all are available.
//the String 'fuck' is to read the last letter '\n' behind the float number b.
